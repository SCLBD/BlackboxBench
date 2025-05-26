from surrogate_model.imagenet_models.adv_resnet50_gelu.ghost_bn import GhostBN2D_ADV
from surrogate_model.imagenet_models.adv_resnet50_gelu.affine import Affine
from torch import nn

class EightBN(nn.Module):
        
    def __init__(self, num_features, *args, virtual2actual_batch_size_ratio=2, affine=False, sync_stats=False, **kwargs):
        super(EightBN, self).__init__()
        virtual2actual_batch_size_ratio = 2
        
        self.bn0 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn1 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn2 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn3 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn4 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn5 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn6 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn7 = GhostBN2D_ADV(num_features = num_features, *args, virtual2actual_batch_size_ratio=virtual2actual_batch_size_ratio, affine=affine, sync_stats=sync_stats, **kwargs)
        self.bn_type = 'bn0'
        self.aff = Affine(width=num_features, k=1)
        
    def forward(self, input):
        if self.bn_type == 'bn0':
            input = self.bn0(input)
        elif self.bn_type == 'bn1':
            input = self.bn1(input)
        elif self.bn_type == 'bn2':
            input = self.bn2(input)
        elif self.bn_type == 'bn3':
            input = self.bn3(input)
        elif self.bn_type == 'bn4':
            input = self.bn4(input)
        elif self.bn_type == 'bn5':
            input = self.bn5(input)
        elif self.bn_type == 'bn6':
            input = self.bn6(input)
        elif self.bn_type == 'bn7':
            input = self.bn7(input)
        
        input = self.aff(input)
        return input