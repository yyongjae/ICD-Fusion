import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool
import torch.nn.functional as F

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-7

    def forward(self, x1, x2):
        sim_matrix = torch.matmul(x1, x2.T) / self.temperature
        sim_matrix = sim_matrix - torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        labels = torch.arange(x1.size(0)).to(x1.device)
        loss = nn.CrossEntropyLoss()((sim_matrix + self.eps), labels)
        return loss

class CDLSSTransform(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channel = self.model_cfg.IN_CHANNEL
        out_channel = self.model_cfg.OUT_CHANNEL
        self.image_size = self.model_cfg.IMAGE_SIZE
        self.feature_size = self.model_cfg.FEATURE_SIZE

        xbound = self.model_cfg.XBOUND
        ybound = self.model_cfg.YBOUND
        zbound = self.model_cfg.ZBOUND
        self.dbound = self.model_cfg.DBOUND

        downsample = self.model_cfg.DOWNSAMPLE

        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channel
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]

        self.dtransform = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )

        self.depthnet = nn.Sequential(
            nn.Conv2d(in_channel + 64, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, in_channel, 3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, self.D + self.C, 1),
        )
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
                nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

        self.forward_ret_dict = {}

    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)
        frustum = torch.stack((xs, ys, ds), -1)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, camera2lidar_rots, camera2lidar_trans, intrins, post_rots, post_trans, **kwargs):
        camera2lidar_rots = camera2lidar_rots.to(torch.float)
        camera2lidar_trans = camera2lidar_trans.to(torch.float)
        intrins = intrins.to(torch.float)
        post_rots = post_rots.to(torch.float)
        post_trans = post_trans.to(torch.float)

        B, _ = camera2lidar_trans.shape

        # undo image-aug
        points = self.frustum - post_trans.view(B, 1, 1, 1, 3)  # (B, D, fH, fW, 3)
        points = torch.inverse(post_rots).view(B, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # (B, D, fH, fW, 3, 1)

        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3], points[:, :, :, :, 2:3]), 4)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)  # (B,D,fH,fW,3)
        points += camera2lidar_trans.view(B, 1, 1, 1, 3)

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = extra_rots.view(B, 1, 1, 1, 3, 3).repeat(1, 1, 1, 1, 1, 1) \
                .matmul(points.unsqueeze(-1)).squeeze(-1)

        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 3).repeat(1, 1, 1, 1, 1)
        return points

    def bev_pool(self, geom_feats, x):
        geom_feats = geom_feats.to(torch.float)
        x = x.to(torch.float)

        B, D, H, W, C = x.shape
        Nprime = B * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)  # (B*D*H*W, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  # (Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])  # (Nprime, 1)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # (Nprime, 4 | X, Y, Z, batch_ix)

        # filter out points that are outside box
        kept = (
                (geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]  # (num_points_valid, 80)
        geom_feats = geom_feats[kept]  # (num_points_valid, 4)
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        # print(x.shape)  [2, 80, 1, 234, 266]
        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)  # (B, C*D, H, W)
        return final

    def get_cam_feats(self, x, d):
        B, C, fH, fW = x.shape

        d = d.view(B, *d.shape[1:])
        x = x.view(B, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

        x = x.view(B, self.C, self.D, fH, fW)  # (B, 80, D, fH, fW)
        x = x.permute(0, 2, 3, 4, 1)  # (B, D, fH, fW, 80)
        return x

    def assign_target(self, gt_boxes):
        x_loc, y_loc, dx, dy = gt_boxes[:, :, 0], gt_boxes[:, :, 1], gt_boxes[:, :, 3], gt_boxes[:, :, 4]

        x_corners = torch.stack([dx / 2., dx / 2., -dx / 2., -dx / 2.], dim=-1).unsqueeze(-1)  # (B, N, 4)
        y_corners = torch.stack([dy / 2., -dy / 2., -dy / 2., dy / 2.], dim=-1).unsqueeze(-1)   # (B, N, 4)

        x = x_loc.unsqueeze(-1).unsqueeze(-1)  + x_corners  # (B, N, 4, 1)
        y = y_loc.unsqueeze(-1).unsqueeze(-1)  + y_corners  # (B, N, 4, 1)

        corners = torch.cat((x, y), dim=-1)
        B, N, _, _ = corners.shape
        corners_flat = corners.view(-1, 4, 2)
        grid_coords_flat = torch.zeros(B * N, 4, 2)

        for corner_idx in range(4):
            x, y = corners_flat[:, corner_idx, 0], corners_flat[:, corner_idx, 1]

            grid_x = ((x - (self.bx[0] - self.dx[0] / 2.0)) / self.dx[0]).floor().long()
            grid_y = ((y - (self.bx[1] - self.dx[1] / 2.0)) / self.dx[1]).floor().long()
            grid_x = torch.clamp(grid_x, 0, self.nx[0] - 1)
            grid_y = torch.clamp(grid_y, 0, self.nx[1] - 1)
            grid_coords_flat[:, corner_idx, 0] = grid_x
            grid_coords_flat[:, corner_idx, 1] = grid_y

        grid_coords = grid_coords_flat.view(B, N, 4, 2)

        for batch_idx in range(B):
            batch_bboxes = grid_coords[batch_idx]  # (N, 4, 2)

            xmin = batch_bboxes[:, :, 0].min(dim=1)[0].long()
            xmax = batch_bboxes[:, :, 0].max(dim=1)[0].long()
            ymin = batch_bboxes[:, :, 1].min(dim=1)[0].long()
            ymax = batch_bboxes[:, :, 1].max(dim=1)[0].long()

            bbox = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
            if batch_idx == 0:
                bbox_tensor = bbox.unsqueeze(0)  # (1, N, 4)
            else:
                bbox_tensor = torch.cat((bbox_tensor, bbox.unsqueeze(0)), dim=0)  # (B, N, 4)
        return bbox_tensor

    def get_loss(self, tb_dict):
        tb_dict = {} if tb_dict is None else tb_dict

        img = self.forward_ret_dict['img_bev'].cuda()
        gt = self.forward_ret_dict['gt_bev']
        bbox_tensor = self.forward_ret_dict['bev_boxes']

        B, C, H, W = img.shape
        output_size = (9, 9)

        pooled_features1_batch = []
        pooled_features2_batch = []
        for batch_idx in range(B):
            batch_bboxes = bbox_tensor[batch_idx]  # (N, 4)
            pooled_features1 = []
            pooled_features2 = []
            for bbox_idx in range(batch_bboxes.shape[0]):
                xmin, ymin, xmax, ymax = batch_bboxes[bbox_idx]
                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, W)
                ymax = min(ymax, H)
                cropped_feature_map1 = img[batch_idx, :, ymin:ymax, xmin:xmax]
                cropped_feature_map2 = gt[batch_idx, :, ymin:ymax, xmin:xmax]
                if cropped_feature_map1.shape[1] > 0 and cropped_feature_map1.shape[2] > 0:
                    pooled_feature1 = F.adaptive_avg_pool2d(cropped_feature_map1, output_size)  # (C1, 9, 9)
                else:
                    pooled_feature1 = torch.zeros((img.shape[1], *output_size)).cuda()

                if cropped_feature_map2.shape[1] > 0 and cropped_feature_map2.shape[2] > 0:
                    pooled_feature2 = F.adaptive_max_pool2d(cropped_feature_map2, output_size)  # (C2, 9, 9)
                else:
                    pooled_feature2 = torch.zeros((gt.shape[1], *output_size)).cuda()
                pooled_features1.append(pooled_feature1.unsqueeze(0)) # (1, C1, 9, 9)
                pooled_features2.append(pooled_feature2.unsqueeze(0))  # (1, C2, 9, 9)

            pooled_features1 = torch.cat(pooled_features1, dim=0)  # (N, C1, 9, 9)
            pooled_features2 = torch.cat(pooled_features2, dim=0)  # (N, C2, 9, 9)
            pooled_features1_batch.append(pooled_features1.unsqueeze(0))  # (1, N, C1, 9, 9)
            pooled_features2_batch.append(pooled_features2.unsqueeze(0))  # (1, N, C2, 9, 9)

        pooled_features1_batch = torch.cat(pooled_features1_batch, dim=0) # (B, N, C1, 9, 9)
        pooled_features2_batch = torch.cat(pooled_features2_batch, dim=0)  # (B, N, C2, 9, 9)
        B, N, _, _, _ = pooled_features1_batch.shape

        gt_embeddings = pooled_features2_batch.view(B*N, 64, 8, 9, 9).max(dim=2)[0]  # max across dim=2
        img_embeddings = pooled_features1_batch.view(B*N, 64, 2, 9, 9).max(dim=2)[0]  # max across dim=2

        bev_embeddings_flat = img_embeddings.reshape(B*N, -1)
        gt_embeddings_flat = gt_embeddings.reshape(B*N, -1)
        bev_embeddings_flat = F.normalize(bev_embeddings_flat, dim=-1)
        gt_embeddings_flat = F.normalize(gt_embeddings_flat, dim=-1)
        contrastive_loss_fn = NTXentLoss(temperature=0.5)
        loss = contrastive_loss_fn(bev_embeddings_flat, gt_embeddings_flat)

        tb_dict.update({'DDLSS_loss': loss.item()})
        return loss, tb_dict

    def forward(self, batch_dict):
        x = batch_dict['image_fpn']
        x = x[0]
        BN, C, H, W = x.size()  # 2, 3, 256, 1024
        img = x.view(BN, C, H, W)

        camera_intrinsics = batch_dict['camera_intrinsics']  # (2,3,3)
        camera2lidar = batch_dict['camera2lidar']  # (2,4,4)
        img_aug_matrix = batch_dict['img_aug_matrix']  # (2,4,4)
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']  # (2,4,4)
        lidar2image = batch_dict['lidar2image']  # (2,3,4)

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        points = batch_dict['points']

        batch_size = BN
        depth = torch.zeros(batch_size, 1, *self.image_size).to(points[0].device)  # (B,1,H,W)
        for b in range(batch_size):
            batch_mask = points[:, 0] == b
            cur_coords = points[batch_mask][:, 1:4]  # num_points,3
            cur_img_aug_matrix = img_aug_matrix[b]  # 4,4
            cur_lidar_aug_matrix = lidar_aug_matrix[b]  # 4,4
            cur_lidar2image = lidar2image[b]  # 3,4

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )  # (3, num_points)

            # lidar2image
            cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)  # (3, num_points)
            # print(cur_coords[:,0])  ---->  [3.9952e+04, 6.9925e+03, 3.7159e+01]
            dist = cur_coords[2, :]
            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]  # (3, num_points)

            # do image aug
            cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:3, 3].reshape(3, 1)  # (3, num_points)
            cur_coords = cur_coords[:2, :].transpose(0, 1)  # (num_points,2| u,v)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]  # (x,y)->(y,x)
            # print(cur_coords[0])  --->  [ 39.8298, 893.6777]
            # filter points outside of images
            on_img = (
                    (cur_coords[..., 0] < self.image_size[0])
                    & (cur_coords[..., 0] >= 0)
                    & (cur_coords[..., 1] < self.image_size[1])
                    & (cur_coords[..., 1] >= 0)
            )  # (num_points,)
            masked_coords = cur_coords[on_img].long()  # (num_points_on_img, 2)

            masked_dist = dist[on_img]
            depth[b, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
            # (B, 1, H, W)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots,
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )  # (B, D, fH, fW, 3)

        # use points depth to assist the depth prediction in images
        x = self.get_cam_feats(img, depth)  # (B, D, fH, fW, 80)

        x = self.bev_pool(geom, x)  # (B, C, H, W)
        x = self.downsample(x)
        x = x.permute(0, 1, 3, 2)
        if self.training:
            gt_boxes = batch_dict['gt_boxes']
            bboxes = self.assign_target(gt_boxes)
            self.forward_ret_dict['gt_bev'] = batch_dict['gt_bev_features']
            self.forward_ret_dict['img_bev'] = x
            self.forward_ret_dict['bev_boxes'] = bboxes
        # convert bev features from (b, c, x, y) to (b, c, y, x)
        batch_dict['spatial_features_img'] = x
        return batch_dict