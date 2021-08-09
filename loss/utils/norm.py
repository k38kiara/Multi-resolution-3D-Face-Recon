import torch

class NormLoss:
    @staticmethod
    def norm(preds: torch.Tensor, 
            gts: torch.Tensor, 
            loss_type: str,
            masks: torch.Tensor = None, 
            ) -> torch.Tensor:

        assert (loss_type in ['l1', 'l2', 'l21']), "Suporting loss type is ['l1', 'l2', 'l21']"
        diff = preds - gts
        if masks is not None:
            diff = torch.mul(diff, masks)

        if loss_type == 'l1':
            loss = torch.mean(torch.abs(diff))
        elif loss_type == 'l2':
            loss = torch.mean(torch.sqrt(diff ** 2))
        elif loss_type == 'l21':
            loss = torch.sqrt(torch.sum(diff**2 + 1e-16, axis = -1))
            loss = torch.sum(loss)
            loss = loss / torch.numel(diff)
        
        return loss