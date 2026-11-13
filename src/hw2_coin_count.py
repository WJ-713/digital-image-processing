# -*- coding: utf-8 -*-
"""
Homework 2 - Coin Counting
硬币计数（数字图像处理 · OpenCV）

思路：
    硬币是深色背景上的亮目标，而且表面有浮雕纹理、光照不均，
    单纯用全局 Otsu 阈值会把偏暗的那枚硬币削掉一大块（只剩个月牙，
    圆度只有 0.35），所以改用自适应阈值；
    1. 灰度化 + 高斯滤波；
    2. 自适应阈值（高斯，blockSize=51，C=-5）——按局部邻域亮度分割，
       浅色币、深色币都能完整取到；
    3. 清掉图片四周的黑边框；
    4. 开运算去掉纹理噪点，闭运算补上币面裂缝；
    5. 孔洞填充，把币面反光造成的空洞补实，得到完整圆盘；
    6. 距离变换 + 分水岭：硬币互相叠压时按距离脊线切开；
    7. 用“面积 + 圆度”双重过滤：
       圆度 = 4πA / P²，越接近 1 越像圆；残留碎片、背景碎块都被剔掉。
       （分水岭边界有锯齿会让周长略微偏大，所以圆度阈值取 0.6 留余量。）
    8. 统计通过过滤的区域个数即为硬币数量，并画出轮廓、编号、半径。

用法：
    python src/hw2_coin_count.py
    python src/hw2_coin_count.py --input data/hw2_coin_2183.png --output results
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# 脚本位于 <repo>/src/ 下，默认按仓库结构读写 data/ 与 results/
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="统计图片中的硬币数量")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "hw2_coin_2183.png"),
                        help="输入图片路径")
    parser.add_argument("--output", default=str(REPO_ROOT / "results"),
                        help="输出目录（存放标注图与掩膜图）")
    parser.add_argument("--min-area", type=int, default=2000,
                        help="有效硬币的最小面积，用于滤掉碎片")
    parser.add_argument("--min-circularity", type=float, default=0.6,
                        help="有效硬币的最小圆度（4πA/P²）")
    parser.add_argument("--border", type=int, default=12,
                        help="清除图片边框的宽度")
    return parser.parse_args()


def clear_border(mask, border):
    """把四周 border 像素宽的边缘清零，去掉图片黑边框。"""
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def fill_holes(binary):
    """从图像外角泛洪填充背景，未填充到的内部空洞即为孔洞，补实。"""
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    return binary | cv2.bitwise_not(flood)


def segment(bgr, border):
    """自适应阈值分割，返回 (二值图, 填洞后的实心前景)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 硬币比局部背景亮，用高斯自适应阈值，适应币面反光和光照不均
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, -5)
    binary = clear_border(binary, border)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel5, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel7, iterations=2)

    filled = fill_holes(binary)
    filled = clear_border(filled, border)
    return binary, filled


def watershed_split(bgr, filled):
    """距离变换 + 分水岭，把相互叠压的硬币切开，返回标记图。"""
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

    # 圆盘内部距离最大处就是币心，按最大距离的 35% 取种子
    _, sure_fg = cv2.threshold(dist, 0.35 * float(dist.max()), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(filled, kernel5, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(bgr.copy(), markers)
    return markers


def count_and_draw(bgr, markers, min_area, min_circularity):
    """按面积与圆度过滤区域，统计个数并在图上绘制轮廓、编号与半径。"""
    result = bgr.copy()
    count = 0
    for label in range(2, markers.max() + 1):
        region = (markers == label).astype(np.uint8)
        area = int(region.sum())
        if area < min_area:
            continue

        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(biggest, True)
        # 圆度：4πA/P²，完美圆为 1，越扁长越接近 0
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity < min_circularity:
            continue

        count += 1
        cv2.drawContours(result, contours, -1, (0, 200, 0), 2)

        (cx, cy), radius = cv2.minEnclosingCircle(biggest)
        cv2.putText(result, str(count), (int(cx) - 10, int(cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(result, "r=%d" % int(radius), (int(cx) - 22, int(cy) + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.putText(result, "count = %d" % count, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return result, count


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("无法读取图片：%s" % input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    binary, filled = segment(bgr, args.border)
    markers = watershed_split(bgr, filled)
    result, count = count_and_draw(bgr, markers, args.min_area, args.min_circularity)

    result_path = output_dir / "hw2_coin_result.png"
    mask_path = output_dir / "hw2_coin_mask.png"
    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(mask_path), filled)

    print("输入图片：%s" % input_path)
    print("图像尺寸：%dx%d" % (binary.shape[1], binary.shape[0]))
    print("硬币数量：%d" % count)
    print("标注图：%s" % result_path)
    print("前景掩膜：%s" % mask_path)


if __name__ == "__main__":
    main()
