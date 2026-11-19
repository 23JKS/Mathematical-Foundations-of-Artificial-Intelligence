from math import sqrt

import torch
from torch import Tensor, nn

from utility import channel_norm, conv1d, deconv1d


class TVRegular(nn.Module):
    def __init__(self, lbd: float, eps: float = 1e-6) -> None:
        '''
        :param lbd: 正则化系数
        :param eps: 数值稳定参数
        '''
        super().__init__()
        self.lbd: float = lbd
        self.eps: float = eps

    def forward(self, z: Tensor) -> Tensor:
        '''
        :param z: 估计图像, (..., C, H, W) 0~1
        :return: 正则化损失项, (...,)
        '''
        dx = torch.cat([
            z[..., 0:1],
            z[..., 1:] - z[..., :-1]
        ], dim=-1)
        dy = torch.cat([
            z[..., 0:1, :],
            z[..., 1:, :] - z[..., :-1, :]
        ], dim=-2)
        grad = (dx**2+dy**2).sum(dim=-3)
        return self.lbd * torch.sqrt(grad+self.eps).sum(dim=[-1, -2])


class MSELoss(nn.Module):
    def forward(self, z: Tensor, y: Tensor) -> Tensor:
        '''
        :param z: 估计图像, (..., C, H, W) 0~1
        :param y: 加噪图像, (..., C, H, W) 0~1
        :return: 损失, (...,)
        '''
        return 0.5 * channel_norm(y-z).norm(p=2, dim=[-1, -2])**2


HAAR_H: Tensor = Tensor([1, 1]) / sqrt(2)
HAAR_G: Tensor = Tensor([1, -1]) / sqrt(2)


class DWT(nn.Module):
    def __init__(self, h: Tensor, g: Tensor,  n: int) -> None:
        '''
        :param h: 低通滤波器, (K,)
        :param g: 高通滤波器, (K,)
        :param n: 滤波层数
        '''
        super().__init__()
        self.h = nn.Parameter(h.view(1, 1, -1), requires_grad=False)  # (1, 1, K)
        self.g = nn.Parameter(g.view(1, 1, -1), requires_grad=False)  # (1, 1, K)
        self.n0: int = h.size(-1) // 2
        self.n: int = n

    def forward(self, y: Tensor) -> Tensor:
        '''
        :param y: 加噪图像, (..., C, H, W) 0~1
        :return: 小波系数, (..., C, H, W)
        '''
        # 每层只继续分解左上角的低频块（LL），避免逐层复制整张系数图。
        blocks = []
        ll = y
        for _ in range(self.n):
            height, width = ll.shape[-2:]
            block = self.dwt(ll)
            blocks.append(block)
            ll = block[..., :height // 2, :width // 2]
        # 从最深层往回拼接，保留各层的高频系数和原来的系数布局。
        for block in reversed(blocks):
            height, width = ll.shape[-2:]
            top = torch.cat((ll, block[..., :height, width:]), dim=-1)
            ll = torch.cat((top, block[..., height:, :]), dim=-2)
        return ll if blocks else y.clone()

    def dwt(self, y: Tensor) -> Tensor:
        '''
        :param y: 加噪图像, (..., C, H, W) 0~1
        :return: 小波系数, (..., C, H, W)
        '''
        height = y.size(-2)
        width = y.size(-1)
        ca = conv1d(y, self.h, dim=-1)[..., self.n0:width+self.n0:2]
        ll = conv1d(ca, self.h, dim=-2)[..., self.n0:height+self.n0:2, :]
        lh = conv1d(ca, self.g, dim=-2)[..., self.n0:height+self.n0:2, :]
        cd = conv1d(y, self.g, dim=-1)[..., self.n0:width+self.n0:2]
        hl = conv1d(cd, self.h, dim=-2)[..., self.n0:height+self.n0:2, :]
        hh = conv1d(cd, self.g, dim=-2)[..., self.n0:height+self.n0:2, :]
        return torch.cat([
            torch.cat([ll, hl], dim=-1),
            torch.cat([lh, hh], dim=-1)
        ], dim=-2)


class IDWT(nn.Module):
    def __init__(self, h: Tensor, g: Tensor, n: int) -> None:
        '''
        :param h: 低通滤波器, (K,)
        :param g: 高通滤波器, (K,)
        :param n: 重构层数，应与 DWT 的 n 一致
        '''
        super().__init__()
        self.h = nn.Parameter(h.view(1, 1, -1), requires_grad=False)
        self.g = nn.Parameter(g.view(1, 1, -1), requires_grad=False)
        self.n0: int = h.size(-1) // 2
        self.n: int = n

    def forward(self, w: Tensor) -> Tensor:
        '''
        :param w: 多级小波系数, (..., C, H, W)
        :return: 重建图像, (..., C, H, W)
        '''
        w = w.clone()
        for level in range(self.n, 0, -1):
            block_h = w.size(-2) // (2 ** (level - 1))
            block_w = w.size(-1) // (2 ** (level - 1))
            w[..., :block_h, :block_w] = self.idwt(
                w[..., :block_h, :block_w]
            )
        return w

    def idwt(self, y: Tensor) -> Tensor:
        '''
        :param y: 单级小波系数, (..., C, H, W)，布局为 [LL, HL; LH, HH]
        :return: 重建图像, (..., C, H, W)
        '''
        height = y.size(-2)
        width = y.size(-1)
        h_half = height // 2
        w_half = width // 2
        ll = y[..., :h_half, :w_half]
        hl = y[..., :h_half, w_half:]
        lh = y[..., h_half:, :w_half]
        hh = y[..., h_half:, w_half:]
        offset = 1 + self.n0
        ca = deconv1d(ll, self.h, dim=-2, l_out=height, offset=offset) + \
            deconv1d(lh, self.g, dim=-2, l_out=height, offset=offset)
        cd = deconv1d(hl, self.h, dim=-2, l_out=height, offset=offset) + \
            deconv1d(hh, self.g, dim=-2, l_out=height, offset=offset)
        out = deconv1d(ca, self.h, dim=-1, l_out=width, offset=offset) + \
            deconv1d(cd, self.g, dim=-1, l_out=width, offset=offset)
        return out
