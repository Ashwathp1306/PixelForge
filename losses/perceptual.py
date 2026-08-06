import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    """
    VGG-19 based Perceptual Feature Loss.
    Extracts features at relu2_2 and relu3_3 layers to compute L1 loss in feature space.
    """
    def __init__(self, feature_layers=(8, 17), weight: float = 1.0):
        super().__init__()
        self.weight = weight
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        
        for x in range(feature_layers[0]):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(feature_layers[0], feature_layers[1]):
            self.slice2.add_module(str(x), vgg[x])
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # If single channel grayscale, expand to 3 channels for VGG
        if pred.size(1) == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        
        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        h_pred1 = self.slice1(pred_norm)
        h_target1 = self.slice1(target_norm)
        
        h_pred2 = self.slice2(h_pred1)
        h_target2 = self.slice2(h_target1)

        loss1 = nn.functional.l1_loss(h_pred1, h_target1)
        loss2 = nn.functional.l1_loss(h_pred2, h_target2)

        return self.weight * (loss1 + loss2)
