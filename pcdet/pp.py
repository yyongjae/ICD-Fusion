import torch
import torch.nn as nn
import torch.nn.functional as F

class BEVContrastiveLearning:
    def __init__(self, pooling_size=(7, 7), logit_scale_init=1.0):
        """
        初始化对比学习模块
        :param pooling_size: 池化大小 (默认全局平均池化)
        :param logit_scale_init: 对比学习的初始 logit 缩放比例
        """
        self.pooling = nn.AdaptiveAvgPool2d(pooling_size)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def crop_and_pool(self, bev_features, boxes):
        """
        从 BEV 特征图中裁剪目标区域并池化。

        :param bev_features: BEV 特征图, Tensor, shape: (B, C, H, W)
        :param boxes: 目标边界框 (x1, y1, x2, y2)，Tensor, shape: (N, 4)
        :return: 池化后的目标特征, Tensor, shape: (N, C)
        """
        pooled_features = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.int()
            cropped = bev_features[:, :, y1:y2, x1:x2]  # 裁剪区域
            pooled = self.pooling(cropped)  # 池化
            pooled_features.append(pooled.squeeze(-1).squeeze(-1))  # 去除多余维度
        return torch.cat(pooled_features, dim=0)

    def compute_similarity_matrix(self, bev_embeddings, gt_embeddings):
        """
        计算 BEV 特征与 GT-BEV 特征之间的相似性矩阵。

        :param bev_embeddings: BEV 特征, Tensor, shape: (N, C)
        :param gt_embeddings: GT-BEV 特征, Tensor, shape: (N, C)
        :return: 相似性矩阵, Tensor, shape: (N, N)
        """
        # 特征归一化
        bev_embeddings = F.normalize(bev_embeddings, p=2, dim=1)
        gt_embeddings = F.normalize(gt_embeddings, p=2, dim=1)

        # 相似性矩阵
        similarity_matrix = torch.matmul(bev_embeddings, gt_embeddings.T) * self.logit_scale.exp()
        return similarity_matrix

    def compute_loss(self, similarity_matrix, target_similarity_matrix):
        """
        计算对比损失。

        :param similarity_matrix: BEV 和 GT-BEV 特征相似性矩阵, shape: (N, N)
        :param target_similarity_matrix: 目标相似性矩阵, shape: (N, N)
        :return: 对比损失
        """
        # 交叉熵损失沿两个轴计算
        loss_1 = F.cross_entropy(similarity_matrix, target_similarity_matrix.argmax(dim=1))
        loss_2 = F.cross_entropy(similarity_matrix.T, target_similarity_matrix.argmax(dim=0))
        loss = (loss_1 + loss_2) / 2
        return loss

    def forward(self, bev_features, gt_features, boxes, target_similarity_matrix):
        """
        执行 BEV 对比学习过程。

        :param bev_features: BEV 特征图, Tensor, shape: (B, C, H, W)
        :param gt_features: GT-BEV 特征, Tensor, shape: (B, C, H, W)
        :param boxes: 目标边界框 (x1, y1, x2, y2)，Tensor, shape: (N, 4)
        :param target_similarity_matrix: 目标相似性矩阵, Tensor, shape: (N, N)
        :return: 对比学习损失
        """
        # 裁剪并池化 BEV 特征和 GT 特征
        bev_embeddings = self.crop_and_pool(bev_features, boxes)
        gt_embeddings = self.crop_and_pool(gt_features, boxes)

        bev_embeddings = bev_embeddings.view(2,-1)
        gt_embeddings = gt_embeddings.view(2,-1)
        # 计算相似性矩阵
        similarity_matrix = self.compute_similarity_matrix(bev_embeddings, gt_embeddings)

        # 计算损失
        loss = self.compute_loss(similarity_matrix, target_similarity_matrix)

        return loss

# 使用示例
# 假设 BEV 特征图和 GT-BEV 特征图为随机张量，边界框和目标相似性矩阵已知。
bev_features = torch.rand(1, 256, 128, 128)  # 示例 BEV 特征图
gt_features = torch.rand(1, 256, 128, 128)  # 示例 GT-BEV 特征图
boxes = torch.tensor([[30, 30, 60, 60], [70, 70, 100, 100]])  # 两个边界框

target_similarity_matrix = torch.eye(2)  # 目标相似性矩阵（两目标相同）

contrastive_module = BEVContrastiveLearning()
loss = contrastive_module.forward(bev_features, gt_features, boxes, target_similarity_matrix)
print("Loss:", loss.item())