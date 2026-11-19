import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import utils
from torchvision.transforms import functional


def load_image(path: str) -> Tensor:
    '''
    :param path: 图像路径
    :return: 图像, (C, H, W) 0~1
    '''
    img = Image.open(path)
    return functional.to_tensor(img)


def save_image(img: Tensor, path: str) -> None:
    '''
    :param img: 图像, (C, H, W) 0~1
    :param path: 图像路径
    '''
    utils.save_image(img, path)


def resize_image(img: Tensor, height_new: int, width_new: int) -> Tensor:
    '''
    :param img: 图像, (..., H, W) 0~1
    :return: 新尺寸图像, (..., H, W) 0~1
    '''
    return functional.resize(img, [height_new, width_new])


def gaussian_noise(img: Tensor, sigma: float) -> Tensor:
    '''
    :param img: 图像, (..., H, W) 0~1
    :param sigma: 噪声标准差
    :return: 加噪图像, (..., H, W) 0~1
    '''
    noise = torch.randn(img.shape, device=img.device) * sigma
    return (img+noise).clamp(0, 1)


def salt_pepper_noise(img: Tensor, p0: float, p1: float) -> Tensor:
    '''
    :param img: 图像, (..., C, H, W) 0~1
    :param p0: 黑点概率
    :param p1: 白点概率
    :return: 加噪图像, (..., C, H, W) 0~1
    '''
    noise_shape = img.shape[:-3] + (1,) + img.shape[-2:]
    noise = torch.zeros(noise_shape, device=img.device)
    mask = torch.rand(noise_shape, device=img.device)
    noise[mask < p0] = -1
    noise[mask >= (1 - p1)] = 1
    return (img+noise).clamp(0, 1)


def channel_norm(img: Tensor) -> Tensor:
    '''
    :param img: 图像, (..., C, H, W) 0~1
    :return: 通道取范图像, (..., H, W)
    '''
    return img.norm(p=2, dim=-3)


def psnr(z: Tensor, x: Tensor) -> Tensor:
    '''
    :param z: 估计图像, (..., C, H, W) 0~1
    :param x: 原始图像, (..., C, H, W) 0~1
    :return: PSNR, (...,)
    '''
    mse = (x-z).norm(p=2, dim=[-1, -2, -3])**2 / (x.size(-1) * x.size(-2) * x.size(-3))
    return -10 * torch.log10(mse)


def conv1d(f: Tensor, h: Tensor, dim: int) -> Tensor:
    '''
    :param f: 序列, (..., L, ...)
    :param h: 滤波器, (1, 1, K)
    :param dim: 卷积维度
    :return: 卷积序列, (..., L+K-1, ...)
    '''
    l = f.size(dim)
    k = h.size(2)
    f_in = f.transpose(dim, -1)  # (..., L)
    shape = f_in.shape[:-1]
    f_in = f_in.reshape(-1, l).unsqueeze(1)  # (B, 1, L)
    f_out = nn.functional.conv1d(  # pylint: disable=E1102
        f_in,
        h,
        padding=k-1
    )  # (B, 1, L+K-1)
    f_out = f_out.squeeze(1)  # (B, L+K-1)
    l_new = f_out.size(-1)
    y = f_out.reshape(*shape, l_new)  # (..., L+K-1)
    y = y.transpose(-1, dim)  # (..., L+K-1, ...)
    return y


def deconv1d(
    x: Tensor,
    h: Tensor,
    dim: int,
    l_out: int,
    offset: int,
) -> Tensor:
    '''
    :param x: 系数序列, (..., L, ...)
    :param h: 综合滤波器, (1, 1, K)
    :param dim: 卷积维度
    :param l_out: 目标输出长度
    :param offset: 上采样起始偏移，应与 DWT 切片的起始位置一致
    :return: 上采样卷积结果, (..., l_out, ...)
    '''
    L = x.size(dim)
    K = h.size(2)
    x_in = x.transpose(dim, -1)  # (..., L)
    shape = x_in.shape[:-1]
    x_in = x_in.reshape(-1, L)  # (B, L)
    L_up = offset + 2 * L + K - 1
    x_up = torch.zeros(
        x_in.size(0), 1, L_up, device=x.device, dtype=x.dtype
    )  # (B, 1, L_up)
    x_up[:, 0, offset:offset + 2 * L:2] = x_in
    y = nn.functional.conv1d(  # pylint: disable=E1102
        x_up,
        h.flip(-1),
        padding=0,
    )  # (B, 1, L_up + K - 1)
    y = y.squeeze(1)  # (B, L_up + K - 1)
    start = K - 1
    y = y[..., start:start + l_out]
    y = y.reshape(*shape, l_out)  # (..., l_out)
    y = y.transpose(-1, dim)  # (..., l_out, ...)
    return y
