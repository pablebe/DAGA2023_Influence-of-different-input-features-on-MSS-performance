from typing import Final, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-12)
        return torch.view_as_complex(torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1))


class CompressedSpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex

    def forward(self, input, target):
        if not(torch.is_complex(input) and torch.is_complex(target)):
            input = torch.view_as_complex(input)
            target = torch.view_as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        loss = F.mse_loss(input_abs, target_abs) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss

class CCMSELoss(nn.Module):
    sl_f: Final[float]

    def __init__(self, loss_mag_fact, loss_phase_fact, loss_gamma):#state: DF, istft: Optional[Istft] = None):
        super().__init__()
        # loss_config = config['loss_config']
        # self.sr = config['fs']
        # self.fft_size = config['winlen']
        # self.nb_df = p.nb_df
        # self.store_losses = False

        # SpectralLoss
        self.sl_fm = loss_mag_fact
        self.sl_fc = loss_phase_fact
        self.sl_gamma = loss_gamma
        self.sl_f = self.sl_fm + self.sl_fc
        self.sl = CompressedSpectralLoss(factor_magnitude=self.sl_fm, factor_complex=self.sl_fc, gamma=self.sl_gamma)

    def forward(
        self,
        clean: Tensor,
        enhanced: Tensor,
    ):
        sl = [torch.zeros((), device=clean.device)]
        sl = self.sl(input=enhanced, target=clean)
        
        return sl