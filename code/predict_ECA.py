"""加载训练好的 CNNP（含 ECA）模型，对图像做 4 子集残差重建预测。

流程：
    1. 按 (i%2, j%2) 把像素分成 4 个互不重叠的奇偶子集；
    2. 对每个子集，用其余像素的 5×5 加权预测值填充该子集，送入网络得到残差；
    3. 预预测值 + 残差 → 裁剪到 0~255 → 回填该子集，得到最终重建图。

运行前必须修改：
    model_path —— 训练脚本保存的 best_model.pkl 路径
    input_dir  —— 待预测图像目录
    output_dir —— 输出目录（结果统一保存为 TIFF，并附带 metadata.json）

依赖：torch, opencv-python, numpy, Pillow
运行：python predict_ECA.py
"""

import json
import logging
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cuda_available = torch.cuda.is_available()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger("predict_ECA")

# 注意：本文件中的 ECA / CNNP 必须与 train_ECA.py 完全一致；
# 修改网络结构时两处要同步，否则 load_state_dict 会报错。


class ECA(nn.Module):
    """高效通道注意力（Efficient Channel Attention）。

    输入 / 输出均为 [B, C, H, W]，输出为输入乘以逐通道注意力权重。
    """

    def __init__(self, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                             # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))   # [B, C, 1] -> [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)            # [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y.expand_as(x)


# ==================== 网络定义（与训练一致）====================
class CNNP(nn.Module):
    """三分支（3/5/7 卷积）+ 15 个残差块 + ECA 的残差重建网络（与训练脚本一致）。

    输入: [B, 1, H, W]，已归一化到 0~1，掩码位置用邻域预预测值填充；
    输出: [B, 1, H, W]，即预预测值的残差（同一量纲）。
    """

    def __init__(self, eca_k_size=3):
        super(CNNP, self).__init__()
        channel = 32

        # 三个不同感受野的分支：3×3 / 5×5 / 7×7，输出特征逐元素相加
        self.conv1_0 = nn.Sequential(
            nn.Conv2d(1, channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, channel, 5, 1, 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, channel, 7, 1, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1)
        )

        # 15 个残差块：3×3 卷积 → BN → LeakyReLU
        # forward 中按 f = block(f) + identity 逐级相加，即标准残差堆叠
        self.blocks = nn.ModuleList()
        for _ in range(15):
            block = nn.Sequential(
                nn.Conv2d(channel, channel, 3, 1, 1),
                nn.BatchNorm2d(channel),
                nn.LeakyReLU(inplace=True)
            )
            self.blocks.append(block)

        # ECA 放在主干特征之后，对通道做重加权
        self.eca = ECA(k_size=eca_k_size)

        # 输出单通道残差图
        self.conv_out = nn.Conv2d(channel, 1, 3, 1, 1)
        # 可学习的全局偏置（标量）
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, x):
        f1 = self.conv1_0(x)
        f2 = self.conv1_1(x)
        f3 = self.conv1_2(x)
        f = f1 + f2 + f3

        identity = f
        for block in self.blocks:
            f = block(f)
            f = f + identity
            identity = f

        f = self.eca(f)

        out = self.conv_out(f) + self.bias
        return out


# ==================== 加载模型 ====================
model = CNNP()
if cuda_available:
    model = model.cuda()
else:
    model = model.to('cpu')
logger.info("设备: %s", "cuda" if cuda_available else "cpu")

# TODO: 必填，训练脚本保存的最佳模型，例如 r"checkpoints/best_model.pkl"
model_path = ""

# 检查点是自定义 dict，显式关闭 weights_only（只加载自己训练生成的文件）
try:
    if cuda_available:
        checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
    else:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['network'])
    model.eval()
    logger.info("模型加载成功: %s", model_path)
except Exception as e:
    logger.exception("模型加载失败: %s", e)
    exit()


# ==================== 工具函数（与训练完全一致）====================
def generate_mask(height, width, n):
    """生成第 n 个子集的掩码 (n = 0,1,2,3)。

    按 (i%2, j%2) 划分，四个子集互不重叠且恰好覆盖整幅图，每个约占 1/4 像素。
    mask == 1 表示该子集（需要重建的像素），mask == 0 表示已知像素。
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for i in range(n // 2, height, 2):
        for j in range(n % 2, width, 2):
            mask[i, j] = 1
    return mask


def generate_masks(height, width):
    """生成四个子集的掩码列表（n = 0,1,2,3，合起来覆盖整幅图）。"""
    masks = []
    for n in range(4):
        masks.append(generate_mask(height, width, n))
    return masks


def compute_preprediction_excluding_self(img, mask, kernel):
    """对 mask == 1 的像素，用其 5×5 邻域内 mask == 0 的像素做加权平均，
    返回与 img 同尺寸的预预测图像（数值范围 0~255）。

    只填充掩码位置，掩码外保持 0。
    注意：这里为保持脚本独立，使用纯 Python 循环实现；大图（如 4K）会较慢，
    可参考 train_ECA.py 用 numba.njit 加速。
    """
    H, W = img.shape
    pred = np.zeros_like(img, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            if mask[i, j] == 1:
                total_weight = 0.0
                weighted_sum = 0.0
                for di in range(-2, 3):
                    for dj in range(-2, 3):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < H and 0 <= nj < W:
                            if mask[ni, nj] == 0:
                                w = kernel[di + 2, dj + 2]
                                weighted_sum += w * img[ni, nj]
                                total_weight += w
                if total_weight > 1e-6:
                    pred[i, j] = weighted_sum / total_weight
                else:
                    # 邻域内没有可用像素（极小图）时保留原值
                    pred[i, j] = img[i, j]
    return pred


# 固定 5×5 加权核（与训练一致；内部按有效邻域权重再归一化后，整体缩放会被抵消）
mykernel = np.array([
    [0, 1, 0, 1, 0],
    [1, 2, 4, 2, 1],
    [0, 4, 0, 4, 0],
    [1, 2, 4, 2, 1],
    [0, 1, 0, 1, 0]
], dtype=np.float64)
mykernel /= np.sum(mykernel)


# ==================== 预测主函数 ====================
def preprocess_image(image_path):
    """读取图像并转为灰度 float64（CV2 实现）。

    注意：Windows 下 cv2.imread 对含中文的路径可能返回 None，
    若遇到可改用 np.fromfile(path, np.uint8) + cv2.imdecode 读取。
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError(f"CV2无法读取图像：{image_path}")
    img_np = np.array(img_gray, dtype=np.float64)
    return img_np


def predict(image_path):
    """对单张图像执行 4 个子集的重建，返回 uint8 灰度图。"""
    # 读取图像
    img = preprocess_image(image_path)

    height, width = img.shape
    masks = generate_masks(height, width)  # 四个子集掩码，恰好覆盖整图

    # 每个子集的预预测只用其他子集的像素
    pre_images = []
    for n in range(4):
        pre = compute_preprediction_excluding_self(img, masks[n], mykernel)
        pre_images.append(pre)

    # 归一化到 0~1，与训练保持一致
    img_tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0
    if cuda_available:
        img_tensor = img_tensor.cuda()

    predicted_img = img.copy().astype(np.float64)  # 在原始图上逐子集回填

    for n in range(4):
        mask = masks[n]
        pre = pre_images[n]

        pre_tensor = torch.FloatTensor(pre).unsqueeze(0).unsqueeze(0) / 255.0
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).unsqueeze(0)
        if cuda_available:
            pre_tensor = pre_tensor.cuda()
            mask_tensor = mask_tensor.cuda()

        # 构造输入：目标子集位置用预预测值，其余位置保留原图真实像素
        input_batch = img_tensor * (1 - mask_tensor) + pre_tensor * mask_tensor

        with torch.no_grad():
            output = model(input_batch)  # 输出为归一化残差
            output_np = output.cpu().numpy()[0, 0]  # 形状 (H, W)

        # 反归一化：预预测值 + 残差 * 255，并裁剪到 0~255
        compensation = output_np * 255.0
        predicted_subset = pre + compensation
        predicted_subset = np.clip(predicted_subset, 0, 255)

        # 只回填当前子集的像素
        predicted_img[mask == 1] = predicted_subset[mask == 1]

    return np.rint(predicted_img).astype(np.uint8)


# ==================== 批量预测接口 ====================
def predict_folder(input_folder, output_folder):
    """批量预测目录下所有支持格式的图像。

    输出：
        与输入同名的 TIFF 文件（统一编码为 TIFF，即使原扩展名是 .jpg / .png）
        metadata.json —— 记录每张输出图的 [宽, 高]
    """
    os.makedirs(output_folder, exist_ok=True)
    logger.info("输入目录: %s", os.path.abspath(input_folder))
    logger.info("输出目录: %s", os.path.abspath(output_folder))

    exts = ('.tif', '.tiff', '.jpg', '.png', '.bmp')
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]
    total = len(filenames)
    if total == 0:
        logger.warning("输入目录中没有找到支持的图像文件（%s）", ", ".join(exts))
    else:
        logger.info("待处理图像: %d 张", total)

    metadata = {}
    for idx, filename in enumerate(filenames, 1):
        input_path = os.path.join(input_folder, filename)
        try:
            logger.info("(%d/%d) 处理: %s", idx, total, filename)
            with Image.open(input_path) as img:
                orig_width, orig_height = img.size

            pred_img = predict(input_path)
            if pred_img is None or pred_img.dtype != np.uint8:
                raise ValueError(f"无效预测结果: {pred_img.dtype}")

            output_path = os.path.join(output_folder, filename)
            with Image.fromarray(pred_img, mode='L') as out_img:
                out_img.save(output_path, format='TIFF')

            logger.info("(%d/%d) 已保存: %s", idx, total, output_path)
            rel_path = os.path.relpath(output_path, output_folder).replace('\\', '/')
            metadata[rel_path] = [orig_width, orig_height]
        except Exception as e:
            logger.exception("(%d/%d) 处理 %s 失败: %s", idx, total, filename, e)

    if metadata:
        metadata_path = os.path.join(output_folder, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info("元数据已保存: %s", metadata_path)

    success = len(metadata)
    logger.info("批量完成: 共 %d 张，成功 %d 张，失败 %d 张", total, success, total - success)


if __name__ == "__main__":
    input_dir = ""   # TODO: 必填，待预测图像目录
    output_dir = ""  # TODO: 必填，输出目录
    predict_folder(input_dir, output_dir)
    logger.info("预测完成")
