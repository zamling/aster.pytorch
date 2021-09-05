from torch import nn
import torch
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        output = self.fn(self.norm(x)) + x
        return output

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, img_size, channels, patch_size, dim, depth, expansion_factor = 4, dropout = 0.):
    height,width = img_size
    assert (width % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = width // patch_size
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c h (s p1) -> b s (h p1 c)', p1 = patch_size),
        nn.Linear(height* patch_size* channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        # Reduce('b n c -> b c', 'mean'),
        # nn.Linear(dim, num_classes)
    )


if __name__=="__main__":
    model = MLPMixer(
        img_size=(32,100),
        channels=3,
        patch_size=4,
        dim=512,
        depth=12
    )
    img = torch.randn((1,3,32,100))
    print(model(img).shape)
    '''
    参数量，实际运算时间
    aster: Resnet MLP
    CRNN: MLP
    MLP+softmax
    
    
    '''
