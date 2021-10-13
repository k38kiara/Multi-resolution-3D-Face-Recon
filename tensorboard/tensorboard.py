from tensorboardX import SummaryWriter
import config

class Tensorboard():
    def __init__(self, ep_name):
        self.ep_name = ep_name
        self.writer = {}
        self.writer['train'] = SummaryWriter(config.TB_PATH + self.ep_name + '_train')
        self.writer['eval'] = SummaryWriter(config.TB_PATH + self.ep_name + '_eval')
    
    def write_data(self, data, epoch, mode):

        for loss_scalar in data['loss']:
            self.writer[mode].add_scalar(loss_scalar, data['loss'][loss_scalar], epoch+1)
        for i, image in enumerate(data['pred_images'][:4]):
            self.writer[mode].add_image('pred/{}'.format(i), data['pred_images'][i].permute(2, 0, 1), epoch+1)
        for i, image in enumerate(data['gt_images'][:4]):
            self.writer[mode].add_image('gt/{}'.format(i), data['gt_images'][i].permute(2, 0, 1), epoch+1)
        for i, image in enumerate(data['conanical_images'][:4]):
            self.writer[mode].add_image('conanical/{}'.format(i), data['conanical_images'][i].permute(2, 0, 1), epoch+1)