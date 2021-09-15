from . import utils
import os
from . import eval
import torch
from torch.optim import Adam, SGD
from dataset import Dataset
from torch.utils.data import DataLoader
from model import Model, Discriminator
from renderer import pytorch3d_render
from loss import Model_Loss
from tensorboard import Tensorboard
from tqdm import tqdm
import config
from torch.optim.lr_scheduler import StepLR

Renderer = pytorch3d_render()

class Trainer():
    def __init__(self, args):
        
        self.dataset = Dataset('300W-LP')
        self.dataloader = DataLoader(self.dataset, 
                                    batch_size=args.batch_size, 
                                    num_workers=0, 
                                    collate_fn=Dataset.collate_func, 
                                    shuffle=args.shuffle)
        self.mean_faces, self.mean_vertices, self.mean_edges, self.gt_faces = Dataset.load_mean_face(config.DATASET_ROOT + 'meandata/')
        
        self.model = Model(args.mesh_number).cuda()
        #self.optimizer = Adam(params=self.model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(0.5, 0.9))
        self.optimizer = Adam(params=[
                        {'params':self.model.encoder.parameters(), 'lr':1e-4,},
                        {'params':self.model.gcns.parameters(), 'lr':1e-4,},
                        {'params':self.model.gcns_color.parameters(), 'lr':1e-4,},
                        {'params':self.model.light_decoder.parameters(), 'lr':1e-4,},
            ], weight_decay=1e-5, betas=(0.5, 0.9))
        
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.5)
        self.D = Discriminator().cuda()
        self.optimizer_D = Adam(params=self.D.parameters(), lr=1e-5, weight_decay=1e-4, betas=(0.5, 0.9))
        
        self.weight = {'chamfer_dist': 10, 
                       'laplacian_smooth': 1,
                       'landmarks_loss': 1,
                       'pixel': 1, 
                       'symmetric': 0.1, 
                       'perceptual': 0.01,
                       }
        
        self.criteria = Model_Loss()
        self.evaluator = eval.Evaluator(args)
        self.tensorboard = Tensorboard(args.ckpt)
        self.ckpt_path = config.CKPT_ROOT + args.ckpt
        self.train_batch = args.train_batch
        self.epoch = args.epoch
        os.makedirs(self.ckpt_path, exist_ok=True)
        

    def _reset_loss(self):
        return {'chamfer_dist': 0.,
                'laplacian_smooth': 0,
                'landmarks_loss': 0.,
                'pixel': 0.,
                'symmetric': 0.,
                'perceptual': 0.,
                }

    def run(self):
        train_min_loss, eval_min_loss = 100., 100.
        for epoch in range(self.epoch):
            progress_bar_train = tqdm(self.dataloader, ascii=True)
            total_loss = self._reset_loss()
            self.model.train()
            self.D.train()
            
            for i, data in enumerate(progress_bar_train):
                loss, align_images, conanical_images = self._train_step(data, epoch)
                total_loss = {key : total_loss[key] + loss[key] for key in total_loss}
                progress_bar_train.set_description('Loss: %.6f' %(loss['chamfer_dist']))
                if i == self.train_batch - 1:
                    break
                
            total_loss = {key : total_loss[key] / self.train_batch for key in total_loss}
            self.tensorboard.write_data(data={'loss':loss, 
                                        'pred_images':torch.clamp(align_images, 0, 1), 
                                        'gt_images':data['image'], 
                                        'conanical_images': torch.clamp(conanical_images, 0, 1)},
                                         epoch=epoch, 
                                         mode='train')

            
            if (epoch+1) % 10 == 0:
                self.model.eval()
                self.D.eval()
                with torch.no_grad():
                    total_loss = self.evaluator.run(self.model, epoch)
                loss_sum = sum([total_loss[key] * self.weight[key] for key in self.weight])
                if loss_sum < eval_min_loss:
                    eval_min_loss = loss_sum
                    torch.save(self.model.state_dict(), self.ckpt_path + '/best_eval_model.pt')

            self.scheduler.step()
            loss_sum = sum([total_loss[key] * self.weight[key] for key in self.weight])
            if loss_sum < train_min_loss:
                train_min_loss = loss_sum
                torch.save(self.model.state_dict(), self.ckpt_path + '/best_train_model.pt')
            torch.save(self.model.state_dict(), self.ckpt_path + '/last_model.pt')

            if (epoch+1) % 50 == 0:
                torch.save(self.model.state_dict(), self.ckpt_path + '/{}.pt'.format(epoch+1))

    def _train_step(self, data, epoch):
        batch_size = len(data['input_image'])
        
        pred_vertices, color, pred_light = self.model(self.mean_vertices, self.mean_edges, data['input_image'].float(), data)

        rendered_images = Renderer.render(vertices=utils.transform_vertices_coord(pred_vertices[2], data), 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2,  
                                            light=pred_light)

        rendered_canonical_images = Renderer.render(vertices=pred_vertices[2]*128, 
                                            faces=self.mean_faces[2],
                                            colors=(color+1)/2
                                           )

        mask = torch.where(rendered_images.detach().cpu() == torch.tensor([0, 0, 0]), torch.tensor(0), torch.tensor(1)).cuda()
        mask = mask * data['mask']
        align_images = torch.mul(rendered_images, mask) + torch.mul(data['image'], 1-mask)

        all_loss = self.criteria.get_model_loss(
                                    output={'vertices': pred_vertices,
                                            'faces': [self.mean_faces[i] for i in range(3)],
                                            'images': align_images,
                                            'canonical_images': rendered_canonical_images,},
                                    target={'vertices': torch.stack(data['vertices']),
                                            'faces': self.gt_faces,
                                            'images': data['image'],
                                            'masks': data['mask'],
                                            'mean_vertices': self.mean_vertices,
                                    })

        # Loss weight adjust
        if epoch > 3:
            self.weight['perceptual'] = 0.01*((epoch // 10)+1)

        loss = sum([all_loss[key] * self.weight[key] for key in self.weight])
        
        self.optimizer_D.zero_grad()
        reals = data['image'].permute(0, 3, 1, 2)
        fakes = align_images.permute(0, 3, 1, 2)
        fakes_detach = fakes.clone().detach()

        fake_loss = self.criteria.get_D_loss(self.D(fakes_detach), is_real=False)
        real_loss = self.criteria.get_D_loss(self.D(reals), is_real=True)
        dis_loss = (fake_loss + real_loss) * 0.001
        dis_loss.backward()
        self.optimizer_D.step()

        self.optimizer.zero_grad()
        fake_real_loss = self.criteria.get_D_loss(self.D(fakes), is_real=True) * 0.01
        loss = loss + fake_real_loss
        loss.backward()
        self.optimizer.step()

        return all_loss, align_images, rendered_canonical_images