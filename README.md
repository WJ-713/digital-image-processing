# 数字图像处理 · 课程编程测试项目

本仓库用于存放《数字图像处理》课程现场编程测试的完整工程，包含源代码、测试数据、结果图与说明文档。

- 仓库主页：https://github.com/WJ-713/digital-image-processing
- 可见性：Public（教师无需登录即可查看代码与工程）

## 目录结构

```
digital-image-processing/
├── src/            # 源代码（各题目/各算法）
├── data/           # 输入测试图片（小体积样本）
├── results/        # 运行输出的结果图
├── docs/           # 实验说明、算法思路、运行记录
├── requirements.txt
└── README.md
```

> 目录会在测试开始后随实际工程同步更新，以最终提交内容为准。

## 运行环境

- Python 3.x
- 依赖见 `requirements.txt`（核心为 numpy、opencv-python、matplotlib）

```bash
pip install -r requirements.txt
```

## 使用方式

```bash
python src/main.py --input data/sample.png --output results/
```

各算法脚本的单独用法见 `docs/` 下的说明。

## 提交说明

- 编程测试当天在本地完成工程后，将全部源码、数据和结果一次性提交并推送到本仓库；
- 提交信息统一使用 `feat: <题目/功能>` 格式，便于教师按提交历史查看解题过程。
