import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = 'tight'

def set_grid(D, num_cells=1, is_random=False):
    """
    Args: 
        D (Tensor): (n, d1, d2) collection of image tensors
        
    Return:
    """
    if type(D) is np.ndarray:
        A = torch.from_numpy(D)
    else:
        A = D
        
    if len(A.shape) == 3:
        (n, d1, d2) = A.shape
        c = 1
    elif len(A.shape) == 4:
        (n, c, d1, d2) = A.shape
    
    if not is_random:
        img_list = [torch.reshape(A[i], (c, d1, d2)) for i in range(num_cells)]
    else:
        img_list = [torch.reshape(A[i], (c, d1, d2)) for i in range(num_cells)]
        
    return make_grid(img_list)

def show(imgs):
    """
    Args:
        imgs (list): list of images in the form of torch.tensor
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])