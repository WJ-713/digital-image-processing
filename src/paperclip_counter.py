# -*- coding: utf-8 -*-
"""
回形针计数（数字图像处理 · OpenCV）

思路：
    1. 灰度化 + 高斯滤波，抑制背景噪声；
    2. Otsu 自适应阈值反二值化，把深色回形针变成白色前景；
    3. 清掉图片四周的黑边框，避免整幅图被边框连成一块；
    4. 形态学开运算去噪点，闭运算把回形针的断线/细缝补上；
    5. 孔洞填充，让每个回形针由“空心细环”变成“实心长条”，
       这样距离变换才能在回形针内部取到最大值，得到稳定的种子点；
    6. 距离变换 + 阈值取种子（sure_fg），膨胀原图取背景（sure_bg），
       两者差集即“未知区域”，交给分水岭算法分割——
       即使两个回形针相互粘连，也能被分水岭按距离脊线切开；
    7. 统计面积大于阈值的分水岭区域个数，即为回形针数量，
       并在图上画出轮廓、编号。

用法：
    python src/paperclip_counter.py
    python src/paperclip_counter.py --input data/2181.png --output results
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# 脚本位于 <repo>/src/ 下，默认按仓库结构读写 data/ 与 results/
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="统计图片中的回形针数量")
    parser.add_argument("--input", default=str(REPO_ROOT / "data" / "2181.png"),
                        help="输入图片路径")
    parser.add_argument("--output", default=str(REPO_ROOT / "results"),
                        help="输出目录（存放标注图与掩膜图）")
    parser.add_argument("--min-area", type=int, default=2000,
                        help="有效目标的最小面积，用于滤掉残留噪点")
    parser.add_argument("--border", type=int, default=10,
                        help="清除图片边框的宽度")
    return parser.parse_args()


def clear_border(mask, border):
    """把四周 border 像素宽的边缘清零，去掉扫描/拍摄留下的黑边框。"""
    mask[:border, :] = 0
    mask[-border:, :] = 0
    mask[:, :border] = 0
    mask[:, -border:] = 0
    return mask


def fill_holes(binary):
    """从图像外角泛洪填充背景，未填充到的内部空洞即为孔洞，与原图取并集补实。"""
    h, w = binary.shape
    flood = binary.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    return binary | cv2.bitwise_not(flood)


def segment(bgr, border):
    """背景/前景分割，返回 (二值图, 填洞后的实心前景)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 回形针比背景暗，用 Otsu 自动求阈值后反二值化
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    binary = clear_border(binary, border)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel3, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel5, iterations=2)

    filled = fill_holes(binary)
    filled = clear_border(filled, border)
    return binary, filled


def watershed_split(bgr, filled):
    """距离变换 + 分水岭，把粘连目标切开，返回标记图。"""
    dist = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

    # 距离变换的局部峰值就是每个回形针的“核心”，按最大距离的 30% 取种子
    _, sure_fg = cv2.threshold(dist, 0.30 * float(dist.max()), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(filled, kernel3, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 种子点编号：0 是分水岭的“未知区域”，背景标 1，目标从 2 开始
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(bgr.copy(), markers)
    return markers


def count_and_draw(bgr, markers, min_area):
    """统计有效区域个数，并在原图上绘制轮廓与编号。"""
    result = bgr.copy()
    count = 0
    for label in range(2, markers.max() + 1):
        region = (markers == label).astype(np.uint8)
        if int(region.sum()) < min_area:
            continue
        count += 1
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

        # 用最小外接圆圆心放编号，位置比重心更居中
        biggest = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(biggest)
        cv2.putText(result, str(count), (int(cx) - 12, int(cy) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)

    cv2.putText(result, "count = %d" % count, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
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
    result, count = count_and_draw(bgr, markers, args.min_area)

    result_path = output_dir / (input_path.stem + "_result.png")
    contour_path = output_dir / (input_path.stem + "_contour.png")
    cv2.imwrite(str(result_path), result)
    cv2.imwrite(str(contour_path), filled)

    print("输入图片：%s" % input_path)
    print("二值图尺寸：%dx%d" % (binary.shape[1], binary.shape[0]))
    print("回形针数量：%d" % count)
    print("标注图：%s" % result_path)
    print("前景掩膜：%s" % contour_path)


if __name__ == "__main__":
    main()
