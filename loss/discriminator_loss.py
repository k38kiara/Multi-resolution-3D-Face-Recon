import torch
import torch.nn as nn

class Discriminator_Loss():


    @staticmethod
    def get_D_loss(input_result, is_real):
        batch_size = input_result.shape[0]
        if is_real:
            return nn.BCELoss()(input_result, torch.zeros(batch_size, 1).cuda())
        else:
            return nn.BCELoss()(input_result, torch.ones(batch_size, 1).cuda())


