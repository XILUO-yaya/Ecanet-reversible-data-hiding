"""基于 ECA 通道注意力的图像残差重建训练脚本。

数据要求：
    训练目录中只放图像文件（建议 512×512 灰度图，其他尺寸会被缩放到 BLOCK_SIZE）。

运行前必须修改：
    DIRECTORY_data —— 训练图像目录
    model_save_dir —— 模型、日志、CSV 的保存目录

依赖：torch, opencv-python, numpy, scikit-learn, numba
运行：python train_ECA.py
"""

import copy
import csv
import logging
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# 多进程下避免 OpenCV 内部线程与 worker 抢占
cv2.setNumThreads(0)

logger = logging.getLogger("train_ECA")


def setup_logging(log_file):
    """日志同时输出到控制台和文件，文件使用 UTF-8 避免中文乱码。"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8'),
        ],
    )


# ==================== 网络定义 ====================
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


class CNNP(nn.Module):
    """三分支（3/5/7 卷积）+ 15 个残差块 + ECA 的残差重建网络。

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


# ==================== 工具函数 ====================
def get_image(file_path):
    """读取灰度图并转为 float64（数值范围 0~255）。"""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img.astype('float64')


def generate_mask(height, width, n):
    """生成第 n 个 2×2 奇偶子集的掩码（n = 0,1,2,3）。

    按 (i%2, j%2) 划分，四个子集互不重叠且恰好覆盖整幅图，每个约占 1/4 像素。
    mask == 1 表示该子集（需要重建的像素），mask == 0 表示已知像素。
    """
    mask = np.zeros((height, width), dtype=np.float32)
    for i in range(n // 2, height, 2):
        for j in range(n % 2, width, 2):
            mask[i, j] = 1
    return mask


@njit
def compute_preprediction_excluding_self(img, mask, kernel):
    """对 mask == 1 的像素，用其 5×5 邻域内 mask == 0 的像素做加权平均，
    得到预预测图（数值范围 0~255，形状与 img 相同）。

    只填充掩码位置，掩码外保持 0。
    注意：内部按有效邻域权重之和再归一化，因此 kernel 的整体缩放不影响结果。
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


# 固定 5×5 加权核（内部按有效邻域权重再归一化后，这里的整体缩放会被抵消）
MYKERNEL = np.array([
    [0, 1, 0, 1, 0],
    [1, 2, 4, 2, 1],
    [0, 4, 0, 4, 0],
    [1, 2, 4, 2, 1],
    [0, 1, 0, 1, 0]
], dtype=np.float64)
MYKERNEL /= np.sum(MYKERNEL)


def append_csv_row(path, header, row):
    """追加一行 CSV；文件不存在或为空时先写入表头。"""
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# ==================== 数据集 ====================
class TrainDataset(Dataset):
    """训练集：整幅图输入 + 每个样本随机取一个奇偶子集作为“缺失”像素。

    返回:
        input_tensor    [1, H, W]，掩码位置用预预测值填充、其余为真实像素（0~1）
        target_residual [1, H, W]，掩码位置的归一化残差 (img - pred) / 255
        mask_tensor     [1, H, W]，0/1 掩码
    """

    def __init__(self, paths, block_size):
        self.paths = paths
        self.block_size = block_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        bs = self.block_size
        img = get_image(self.paths[idx])
        if img.shape != (bs, bs):
            img = cv2.resize(img, (bs, bs))
        img_block = img.astype(np.float32)

        # 随机选择一个子集作为缺失像素（n = 0,1,2,3）
        n_target = random.randint(0, 3)
        mask_np = generate_mask(bs, bs, n_target)

        # 用固定核计算预预测值
        pred_img = compute_preprediction_excluding_self(img_block, mask_np, MYKERNEL)

        img_tensor = torch.from_numpy(img_block).unsqueeze(0).unsqueeze(0).float() / 255.0
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()

        # 输入：非掩码位置用真实像素，掩码位置用固定核预测值
        input_tensor = img_tensor * (1 - mask_tensor) + pred_tensor * mask_tensor
        # 目标（残差）：仅掩码位置的真实残差
        target_residual = (img_tensor - pred_tensor) * mask_tensor

        return input_tensor, target_residual, mask_tensor


class ValDataset(Dataset):
    """验证集：每张图分别用 n = 0,1,2,3 四个固定子集各评测一次。"""

    def __init__(self, paths, block_size, n_targets=(0, 1, 2, 3)):
        self.items = [(p, n) for n in n_targets for p in paths]
        self.block_size = block_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, n_target = self.items[idx]
        bs = self.block_size
        img = get_image(path)
        if img.shape != (bs, bs):
            img = cv2.resize(img, (bs, bs))
        img_block = img.astype(np.float32)

        mask_np = generate_mask(bs, bs, n_target)
        pred_img = compute_preprediction_excluding_self(img_block, mask_np, MYKERNEL)

        img_tensor = torch.from_numpy(img_block).unsqueeze(0).unsqueeze(0).float() / 255.0
        pred_tensor = torch.from_numpy(pred_img).unsqueeze(0).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()

        input_tensor = img_tensor * (1 - mask_tensor) + pred_tensor * mask_tensor
        target_residual = (img_tensor - pred_tensor) * mask_tensor

        return input_tensor, target_residual, mask_tensor


# ==================== 主流程 ====================
def main():
    # ==================== 超参数 ====================
    LR = 0.001                 # 初始学习率
    BATCH_SIZE = 8             # 显存不足时调小
    WEIGHT_DECAY = 0.001       # AdamW 权重衰减
    EPOCHS = 200               # 训练轮数
    BLOCK_SIZE = 512           # 训练/验证统一缩放的边长
    NUM_WORKERS = 4            # 数据加载进程数（Windows 依赖文件末尾的 __main__ 保护）
    begin_epoch = 0            # 起始轮次（当前未实现断点恢复，保持 0 即可）
    name = "epoch_"            # 定期检查点的文件名前缀
    model_save_dir = ""        # TODO: 必填，模型/日志/CSV 的保存目录

    os.makedirs(model_save_dir, exist_ok=True)
    setup_logging(os.path.join(model_save_dir, 'train.log'))
    logger.info("日志文件: %s", os.path.join(model_save_dir, 'train.log'))

    # ==================== 数据准备 ====================
    DIRECTORY_data = ""  # TODO: 必填，训练图像目录（该目录中应只包含图片文件）
    files_all_path = [os.path.join(DIRECTORY_data, f) for f in sorted(os.listdir(DIRECTORY_data))]

    # 与论文一致：90/10 划分（3000 张时即 2700 训练 / 300 验证）
    files_train_path, files_val_path = train_test_split(
        files_all_path, test_size=0.1, random_state=42
    )
    logger.info("训练集 %d 张，验证集 %d 张", len(files_train_path), len(files_val_path))

    # ==================== 设备与模型 ====================
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    pin_memory = cuda_available
    logger.info("设备: %s（CUDA 可用: %s）", device, cuda_available)
    logger.info("超参数: lr=%s, batch_size=%d, weight_decay=%s, epochs=%d, block_size=%d, workers=%d",
                LR, BATCH_SIZE, WEIGHT_DECAY, EPOCHS, BLOCK_SIZE, NUM_WORKERS)

    train_loader = DataLoader(
        TrainDataset(files_train_path, BLOCK_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        ValDataset(files_val_path, BLOCK_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    mycnn = CNNP().to(device)
    logger.info("模型: CNNP（三分支 + 15 残差块 + ECA，残差学习），参数量 %d",
                sum(p.numel() for p in mycnn.parameters()))

    optimizer = torch.optim.AdamW(mycnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    loss_function = nn.MSELoss()

    # 学习率调度器（每 50 个 epoch 减半，与论文一致）
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # ==================== 训练 ====================
    # 逐 epoch 记录损失，便于后续绘制曲线（本脚本同时写入 CSV）
    train_loss_all = []
    val_loss_all = []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(mycnn.state_dict())

    since = time.time()

    for epoch in range(begin_epoch + 1, EPOCHS + 1):
        logger.info("-" * 60)
        logger.info("Epoch %d/%d | lr=%.6g", epoch, EPOCHS, optimizer.param_groups[0]['lr'])

        # ===== 训练阶段 =====
        mycnn.train()
        train_loss = 0.0
        train_num = 0

        for batch_input, batch_target_residual, batch_mask in train_loader:
            batch_input = batch_input.to(device, non_blocking=True)
            batch_target_residual = batch_target_residual.to(device, non_blocking=True)
            batch_mask = batch_mask.to(device, non_blocking=True)

            output = mycnn(batch_input)
            # 只在掩码位置监督：预测乘掩码，目标残差在掩码外本就是 0。
            # 注意损失以整幅图 H*W 为分母，仅约 1/4 像素有监督，
            # 因此 loss 数值约为掩码位置 MSE 的 1/4。
            loss = loss_function(output * batch_mask, batch_target_residual)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_input.size(0)
            train_num += batch_input.size(0)

        train_loss_avg = train_loss / train_num
        train_loss_all.append(train_loss_avg)
        logger.info("Epoch %d | train_loss=%.8f", epoch, train_loss_avg)

        append_csv_row(
            os.path.join(model_save_dir, 'trainloss.csv'),
            ['epoch', 'train_loss'],
            [epoch, f'{train_loss_avg:.8f}'],
        )

        # ===== 验证阶段（固定遍历四个子集 n = 0,1,2,3） =====
        mycnn.eval()
        val_loss = 0.0
        val_num = 0

        with torch.no_grad():
            for batch_input, batch_target_residual, batch_mask in val_loader:
                batch_input = batch_input.to(device, non_blocking=True)
                batch_target_residual = batch_target_residual.to(device, non_blocking=True)
                batch_mask = batch_mask.to(device, non_blocking=True)

                output = mycnn(batch_input)
                loss = loss_function(output * batch_mask, batch_target_residual)

                val_loss += loss.item() * batch_input.size(0)
                val_num += batch_input.size(0)

        val_loss_avg = val_loss / val_num
        val_loss_all.append(val_loss_avg)
        logger.info("Epoch %d | val_loss=%.8f", epoch, val_loss_avg)

        append_csv_row(
            os.path.join(model_save_dir, 'valloss.csv'),
            ['epoch', 'val_loss'],
            [epoch, f'{val_loss_avg:.8f}'],
        )

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_model_wts = copy.deepcopy(mycnn.state_dict())
            best_model_path = os.path.join(model_save_dir, 'best_model.pkl')
            torch.save({
                'network': best_model_wts,
                'epoch': epoch,
                'val_loss': best_val_loss
            }, best_model_path)
            logger.info("最佳模型已更新: epoch=%d, val_loss=%.8f, 保存至 %s",
                        epoch, best_val_loss, best_model_path)

        # 定期保存（每 5 个 epoch）
        if epoch % 5 == 0:
            checkpoint = {
                'network': mycnn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss_avg,
                'val_loss': val_loss_avg
            }
            ckpt_path = os.path.join(model_save_dir, f'{name}{epoch}.pkl')
            torch.save(checkpoint, ckpt_path)
            logger.info("检查点已保存: %s", ckpt_path)

        elapsed = time.time() - since
        logger.info("累计耗时: %dm%02ds", int(elapsed // 60), int(elapsed % 60))

    logger.info("训练完成，总耗时 %.1f 分钟，最佳验证损失 %.8f",
                (time.time() - since) / 60.0, best_val_loss)


if __name__ == '__main__':
    main()
