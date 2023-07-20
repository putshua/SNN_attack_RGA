import torch
from torchattacks.attack import Attack


class GN(Attack):
    r"""
    Add Gaussian Noise.
    altered from torchattack

    eps = std
    """
    def __init__(self, model, forward_function=None, eps=0.1, T=None, **kwargs):
        super().__init__("GN", model)
        self.eps = eps
        self._supported_mode = ['default']
        self.forward_function = forward_function
        self.T = T

    def forward(self, images, labels=None):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)

        adv_images = images + self.eps*torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images