# -*- coding: utf-8 -*-
"""
Homework 3 - Hole Counting
孔洞检测与计数（数字图像处理 · OpenCV）

任务：
    在 2182.png 这幅法兰盘（金属件）特写中检测四个安装孔，
    并按参考标注图（data/hw3_hole_reference.jpg）的编号方式，
    从上到下依次标为 1~4 并统计数量。

难点：
    画面里除了四个目标孔，还有：右侧一个直径很大的中心大孔、
    左上/左下两块处于边框的暗背景、以及金属表面的阴影和刻字。
    它们的灰度与孔洞相近，只靠阈值会把它们一起算进来。

思路：
    1. 灰度化 + 5×5 高斯滤波；
    2. 自适应阈值反二值化（高斯，blockSize=51，C=8）——
       孔洞比周围金属暗，按局部邻域取阈值可以抵抗画面整体明暗不均（暗角）；
    3. 清掉四周 6 像素边框，直接把贴边的暗背景区域切掉；
    4. 开运算去掉刻字、噪点等小碎块，闭运算补齐孔缘被高光打断的缺口；
    5. 孔洞填充，让孔变成实心圆盘，便于算面积和圆度；
    6. 用三重条件筛选候选孔：
       · 面积在 [min_area, max_area] 内——滤掉碎屑，也把中心大孔排除；
       · 圆度 4πA/P² ≥ 0.6——孔是圆的，阴影、刻字、刻线都不圆；
       · 不外接图像边界——排除边缘暗背景；
    7. 按质心纵坐标从上到下排序后编号（与参考标注图一致），输出数量。

用法：
    python src/hw3_hole_count.py
    python src/hw3_hole_count.py --input data/hw3_hole_2182.png --output results
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# 脚本位于 <repo>/src/ 下，默认按仓库结构读写 data/ 与 results/
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="检测并统计图片中的安装孔数量")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "hw3_hole_2182.png"),
                        help="输入图片路径")
    parser.add_argument("--output", default=str(REPO_ROOT / "results"),
                        help="输出目录（存放标注图与掩膜图）")
    parser.add_argument("--min-area", type=int, default=500,
                        help="候选孔的最小面积，滤掉碎屑与刻字")
    parser.add_argument("--max-area", type=int, default=9000,
                        help="候选孔的最大面积，排除中心大孔")
    parser.add_argument("--min-circularity", type=float, default=0.6,
                        help="候选孔的最小圆度（4πA/P²）")
    parser.add_argument("--border", type=int, default=6,
                        help="清除图像边框的宽度")
    return parser.parse_args()


def clear_border(mask, border):
    """把四周 border 像素宽的边缘清零，切掉贴边的暗背景。"""
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
    """自适应阈值分割暗孔，返回 (二值图, 填洞后的实心前景)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 孔洞比周围金属暗，反二值化后孔变成白色前景
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 51, 8)
    binary = clear_border(binary, border)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=3)

    filled = fill_holes(binary)
    filled = clear_border(filled, border)
    return binary, filled


def find_holes(filled, img_shape, min_area, max_area, min_circularity):
    """按面积、圆度、是否贴边筛选候选孔，返回按从上到下排序的孔列表。"""
    h, w = img_shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(filled, 8)

    holes = []
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue

        x, y = int(stats[label, cv2.CC_STAT_LEFT]), int(stats[label, cv2.CC_STAT_TOP])
        bw, bh = int(stats[label, cv2.CC_STAT_WIDTH]), int(stats[label, cv2.CC_STAT_HEIGHT])
        # 外接图像边界的一律不要（暗背景、画面外的孔）
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
            continue

        region = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        if circularity < min_circularity:
            continue

        cx, cy = centroids[label]
        radius = np.sqrt(area / np.pi)
        holes.append({"center": (float(cx), float(cy)), "radius": float(radius),
                      "area": area, "circularity": float(circularity),
                      "contour": contour})

    # 与参考标注图一致：从上到下（纵坐标小的在前）依次编号 1、2、3……
    holes.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return holes


def draw(result, holes):
    """画出每个孔的轮廓、圆心、编号和半径。"""
    for index, hole in enumerate(holes, start=1):
        contour = hole["contour"]
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        cx, cy = int(cx), int(cy)
        cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
        cv2.line(result, (cx - hole_arm(int(hole["radius"])), cy),
                 (cx + hole_arm(int(hole["radius"])), cy), (0, 255, 255), 1)
        cv2.line(result, (cx, cy - hole_arm(int(hole["radius"]))),
                 (cx, cy + hole_arm(int(hole["radius"]))), (0, 255, 255), 1)

        # 编号放在孔的左上侧，避开孔内的暗区，读起来更清楚
        cv2.putText(result, str(index), (cx - int(hole["radius"]) - 30, cy - int(hole["radius"]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(result, "r=%d" % int(hole["radius"]), (cx - int(hole["radius"]), cy + int(hole["radius"]) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return result


def hole_arm(radius):
    """十字线的臂长，取半径的一半，避免压住孔缘。"""
    return max(4, radius // 2)


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("无法读取图片：%s" % input_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    binary, filled = segment(bgr, args.border)
    holes = find_holes(filled, bgr.shape, args.min_area, args.max_area,
                       args.min_circularity)

    result = draw(bgr.copy(), holes)
    cv2.putText(result, "count = %d" % len(holes), (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    result_path = output_dir / "hw3_hole_result.png"
    mask_path = output_dir / "hw3_hole_mask.png"
    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(mask_path), filled)

    print("输入图片：%s" % input_path)
    print("图像尺寸：%dx%d" % (binary.shape[1], binary.shape[0]))
    for index, hole in enumerate(holes, start=1):
        print("孔%d：圆心(%.0f, %.0f) 半径%.1fpx 面积%d 圆度%.2f"
              % (index, hole["center"][0], hole["center"][1],
                 hole["radius"], hole["area"], hole["circularity"]))
    print("孔的数量：%d" % len(holes))
    print("标注图：%s" % result_path)
    print("前景掩膜：%s" % mask_path)


if __name__ == "__main__":
    main()
