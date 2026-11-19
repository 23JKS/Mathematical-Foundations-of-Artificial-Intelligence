import torch
from pyiqa.archs.gmsd_arch import gmsd
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F
from torch.optim import AdamW
from torchvision.transforms import functional as VF

from model import DWT, HAAR_G, HAAR_H, IDWT, MSELoss, TVRegular
from utility import gaussian_noise, load_image, psnr, save_image
from time import perf_counter
x = load_image(globals().get('image_path', r'./data/test.jpg')).cuda()
y = gaussian_noise(x, 0.2)
print(f'PSNR={psnr(y, x)}, GMSD={gmsd(y.unsqueeze(0), x.unsqueeze(0))}')
save_image(y, r'./data/in.jpg')

dwt = DWT(HAAR_H, HAAR_G, 12).cuda()
idwt = IDWT(HAAR_H, HAAR_G, 12).cuda()
tv_regular = TVRegular(0.2).cuda()
mse_loss = MSELoss().cuda()


def dwt_denoise(y: Tensor, dwt: DWT, idwt: IDWT, thresh: float | Tensor) -> Tensor:
    w = dwt.forward(y)
    w = torch.sign(w) * torch.relu(torch.abs(w) - thresh)
    return idwt.forward(w)


smooth_kernel = Tensor([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]).cuda().view(1, 1, 3, 3) / 20


def smooth_denoise(y: Tensor) -> Tensor:
    return torch.cat([
        F.conv2d(y[0, ...].unsqueeze(0), smooth_kernel, padding=1),  # pylint: disable=E1102
        F.conv2d(y[1, ...].unsqueeze(0), smooth_kernel, padding=1),  # pylint: disable=E1102
        F.conv2d(y[2, ...].unsqueeze(0), smooth_kernel, padding=1)  # pylint: disable=E1102
    ], dim=0)


main = 6
if main == 1:  # TV正则化+空域估计
    shape = [y.size(-2), y.size(-1)]
    y = VF.resize(y, [4096, 4096])
    z = Parameter(y.clone()).cuda()
    opt = AdamW([z])
    for _ in range(100):
        opt.zero_grad()
        loss = mse_loss.forward(z, y) + tv_regular.forward(z)
        loss.backward()
        opt.step()
    z = VF.resize(z, shape)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')
elif main == 2:  # TV正则化+小波域估计
    shape = [y.size(-2), y.size(-1)]
    y = VF.resize(y, [4096, 4096])
    w = Parameter(dwt.forward(y)).cuda()
    opt = AdamW([w])
    for _ in range(100):
        opt.zero_grad()
        z = idwt.forward(w)
        loss = mse_loss.forward(z, y) + tv_regular.forward(z)
        loss.backward()
        opt.step()
    z = idwt.forward(w)
    z = VF.resize(z, shape)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')
elif main == 3:  # 小波阈值去噪
    shape = [y.size(-2), y.size(-1)]
    y = VF.resize(y, [4096, 4096])
    ##可调节
    thresh = 0.01
    z = dwt_denoise(y, dwt, idwt, thresh)
    z = VF.resize(z, shape)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')
elif main == 4:  # 平滑去噪
    z = smooth_denoise(y)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')
# elif main == 5:  # TV正则化+小波阈值学习
#     shape = [y.size(-2), y.size(-1)]
#     y = VF.resize(y, [4096, 4096])
#     thresh = Parameter(torch.zeros(1).cuda())
#     opt = AdamW([thresh])
#     for step in range(100):
#         opt.zero_grad()
#         z = dwt_denoise(y, dwt, idwt, thresh)
#         loss = mse_loss.forward(z, y) + tv_regular.forward(z)
#         loss.backward()
#         opt.step()
#         if step == 1 or step % 10 == 0:
#             print(f"  step {step}/100, loss={loss.item():.4f}", flush=True)
#     print(f'thresh={thresh}')
#     z = dwt_denoise(y, dwt, idwt, thresh)
#     z = VF.resize(z, shape)
#     print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
#     save_image(z, r'./data/out.jpg')

elif main == 5:  # TV正则化+小波阈值学习，修改版优化速度
    shape = [y.size(-2), y.size(-1)]
    y = VF.resize(y, [4096, 4096])
    # 带噪图 y 和小波滤波器在迭代中都不变，只有阈值 thresh 会更新。
    # 缓存原始小波系数，将 DWT 从 101 次（100 轮迭代 + 最终重建）降为 1 次。
    # 每轮仍用当前阈值做软阈值处理和 IDWT，保留对 thresh 的梯度。
    w = dwt.forward(y)
    thresh = Parameter(torch.zeros(1).cuda())
    # opt = AdamW([thresh])
    opt = AdamW([thresh], lr=0.005)
    epochs=50
    # for step in range(epochs):
    #     opt.zero_grad()
    #     z = idwt.forward(torch.sign(w) * torch.relu(torch.abs(w) - thresh))
    #     loss = mse_loss.forward(z, y) + tv_regular.forward(z)
    #     loss.backward()
    #     opt.step()
    #     if step == 1 or step % 10 == 0:
    #         print(f"  step {step}/100, loss={loss.item():.4f}", flush=True)
    #     epochs = 200
    #在每 20、50、100 次迭代时，输出当前阈值、PSNR、GMSD 和累计耗时。
    checkpoints = {20, 40}
    elapsed = 0.0

    torch.cuda.synchronize()
    segment_start = perf_counter()

    for step in range(1, epochs + 1):
        opt.zero_grad()
        # 对固定的小波系数 w 做软阈值处理：小系数归零，大系数保留符号但缩小幅度，再用 IDWT 重建 z。
        z = idwt.forward(torch.sign(w) * torch.relu(torch.abs(w) - thresh))
        loss = mse_loss.forward(z, y) + tv_regular.forward(z)
        loss.backward()
        opt.step()

        if step == 1 or step % 10 == 0:
            print(f"  step {step}/{epochs}, 更新前 loss={loss.item():.4f}", flush=True)

        if step in checkpoints:
            torch.cuda.synchronize()
            elapsed += perf_counter() - segment_start

            with torch.no_grad():
                estimate = idwt.forward(torch.sign(w) * torch.relu(torch.abs(w) - thresh))
                estimate = VF.resize(estimate, shape)
                psnr_value = psnr(estimate, x).item()
                gmsd_value = gmsd(estimate.unsqueeze(0), x.unsqueeze(0)).item()

            print(
                f"  [{step} 次更新] PSNR={psnr_value:.4f} dB, "
                f"GMSD={gmsd_value:.6f}, 累计耗时={elapsed:.2f} s",
                flush=True,
            )
            segment_start = perf_counter()
    print(f'thresh={thresh}')
    z = idwt.forward(torch.sign(w) * torch.relu(torch.abs(w) - thresh))
    z = VF.resize(z, shape)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')

elif main == 6:  # TV正则化+小波稀疏正则化
    shape = [y.size(-2), y.size(-1)]
    y = VF.resize(y, [4096, 4096])
    z = Parameter(y.clone()).cuda()
    opt = AdamW([z],lr=0.005)
    epochs=50
    for step in range(epochs):
        opt.zero_grad()
        w = dwt.forward(z)
        ##可调节
        lbd = 0.1
        loss = mse_loss.forward(z, y) + tv_regular.forward(z) + lbd * torch.norm(w, p=1, dim=[-1, -2, -3])
        loss.backward()
        opt.step()
        if step == 1 or step % 10 == 0:
            print(f"  step {step}/{epochs}, loss={loss.item():.4f}", flush=True)
    z = VF.resize(z, shape)
    print(f'PSNR={psnr(z, x)}, GMSD={gmsd(z.unsqueeze(0), x.unsqueeze(0))}')
    save_image(z, r'./data/out.jpg')
