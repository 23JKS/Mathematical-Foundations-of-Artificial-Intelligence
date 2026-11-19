import re
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import torch
from pyiqa.archs.gmsd_arch import gmsd

from utility import psnr

METHODS = (
    "TV 正则化 + 空域估计",
    "TV 正则化 + 小波域估计",
    "固定小波软阈值去噪",
    "卷积平滑去噪",
    "TV 正则化 + 小波阈值学习",
    "TV 正则化 + 小波稀疏正则化",
)


def main():
    script = Path(__file__).with_name("test.py")
    paths = sorted((script.parent / "data" / "CBSD68").glob("*.png"))
    if len(paths) != 68:
        raise RuntimeError(f"CBSD68 应有 68 张 PNG，实际找到 {len(paths)} 张。")

    source = script.read_text(encoding="utf-8")
    selector = re.compile(r"^main[ \t]*=[ \t]*\d+[ \t]*$", re.MULTILINE)
    source, metric_count = re.subn(
        r"""^([ \t]*)print\(f['"]PSNR=\{psnr\(([yz]), x\)[^\n]*""",
        lambda match: f"{match[1]}record({match[2]!r}, {match[2]}, x)",
        source,
        flags=re.MULTILINE,
    )
    if len(selector.findall(source)) != 1 or metric_count != 7:
        raise RuntimeError("test.py 的方案选择或指标输出已变更，请检查脚本。")

    names = ("带噪图基线",) + METHODS
    scores = [[] for _ in names]
    runtimes = [[] for _ in names]
    compiled = [
        compile(selector.sub(f"main = {number}", source), str(script), "exec")
        for number in range(1, len(names))
    ]

    @torch.no_grad()
    def record(kind, image, reference):
        if kind == "y" and number != 1:
            return
        scores[0 if kind == "y" else number].append((
            psnr(image, reference).item(),
            gmsd(image.unsqueeze(0), reference.unsqueeze(0)).item(),
        ))

    with patch("utility.save_image"):
        for index, path in enumerate(paths, start=1):
            for number, code in enumerate(compiled, start=1):
                print(f"[{index}/{len(paths)}] {path.name} | 方法 {number}: {names[number]}", flush=True)
                torch.manual_seed(41 + index)
                torch.cuda.synchronize()
                started = perf_counter()
                exec(code, {"__name__": "__main__", "record": record, "image_path": str(path)})
                torch.cuda.synchronize()
                runtimes[number].append(perf_counter() - started)

    print(f"\nCBSD68: {len(paths)} 张图像")
    print("| 编号 | 方法 | 平均 PSNR (dB) | 平均 GMSD | 平均单图耗时 (s) |")
    print("|---|---|---:|---:|---:|")
    for number, name in enumerate(names):
        if len(scores[number]) != len(paths):
            raise RuntimeError(f"{name} 的指标数量与图像数不一致。")
        psnr_values, gmsd_values = zip(*scores[number])
        time_text = f"{mean(runtimes[number]):.2f}" if number else "-"
        print(f"| {number or '-'} | {name} | {mean(psnr_values):.4f} | "
              f"{mean(gmsd_values):.6f} | {time_text} |")


if __name__ == "__main__":
    main()