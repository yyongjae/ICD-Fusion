import torch
from torch import nn
from pcdet.ops.bev_pool import bev_pool


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class DepthLSSTransform(nn.Module):
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
    
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        # 创建深度张量，维度为D x H x W，D由DBOUND指定，H和W由IMAGE_SIZE指定。
        ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        
        # 按照特征图的尺寸分割图片，并记录它们在图片中的位置
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

        # undo post-transformation
        points = self.frustum - post_trans.view(B, 1, 1, 1, 3)      # (B, D, fH, fW, 3)
        points = torch.inverse(post_rots).view(B, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
        # (B, D, fH, fW, 3, 1)

        # cam_to_lidar
        points = torch.cat((points[:, :, :, :, :2] * points[:, :, :, :, 2:3], points[:, :, :, :, 2:3]), 4)
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)    # (B,D,fH,fW,3)
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
        x = x.reshape(Nprime, C)    # (B*D*H*W, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)     # (Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])      # (Nprime, 1)
        geom_feats = torch.cat((geom_feats, batch_ix), 1)   # (Nprime, 4 | X, Y, Z, batch_ix)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]                         # (num_points_valid, 80)     
        geom_feats = geom_feats[kept]       # (num_points_valid, 4)
        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])
        # print(x.shape)  [2, 80, 1, 234, 266]
        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)   # (B, C*D, H, W)
        return final

    def get_cam_feats(self, x, d):
        '''
        Args:
            x: img, (B, C, fH, fW)
            d: depth, (B, 1, H, W)
        Returns:
            x: (B, D, fH, fW, 80)
        '''
        B, C, fH, fW = x.shape

        d = d.view(B, *d.shape[1:])
        x = x.view(B, C, fH, fW)

        d = self.dtransform(d)
        x = torch.cat([d, x], dim=1)
        x = self.depthnet(x)

        depth = x[:, : self.D].softmax(dim=1)
        x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        x = x.view(B, self.C, self.D, fH, fW)       # (B, 80, D, fH, fW)
        x = x.permute(0, 2, 3, 4, 1)        # (B, D, fH, fW, 80)
        return x

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                image_fpn (list[tensor]): image features after image neck

        Returns:
            batch_dict:
                spatial_features_img (tensor): bev features from image modality
        """
        x = batch_dict['image_fpn'] 
        x = x[0]
        BN, C, H, W = x.size()      # 2, 3, 256, 1024
        img = x.view(BN, C, H, W)

        camera_intrinsics = batch_dict['camera_intrinsics']     # (2,3,3)
        camera2lidar = batch_dict['camera2lidar']       # (2,4,4)
        img_aug_matrix = batch_dict['img_aug_matrix']       # (2,4,4)
        lidar_aug_matrix = batch_dict['lidar_aug_matrix']   # (2,4,4)
        lidar2image = batch_dict['lidar2image']     # (2,3,4)

        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        points = batch_dict['points']

        batch_size = BN
        depth = torch.zeros(batch_size, 1, *self.image_size).to(points[0].device)     # (B,C,1,H,W)

        for b in range(batch_size):
            batch_mask = points[:,0] == b
            cur_coords = points[batch_mask][:, 1:4]     # num_points,3
            cur_img_aug_matrix = img_aug_matrix[b]      # 4,4
            cur_lidar_aug_matrix = lidar_aug_matrix[b]  # 4,4
            cur_lidar2image = lidar2image[b]            # 3,4

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]

            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(
                cur_coords.transpose(1, 0)
            )           # (3, num_points)

            # lidar2image
            cur_coords = cur_lidar2image[:3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:3, 3].reshape(3, 1)      # (3, num_points)
            # print(cur_coords[:,0])  ---->  [3.9952e+04, 6.9925e+03, 3.7159e+01]
            dist = cur_coords[2, :]
            cur_coords[2, :] = torch.clamp(cur_coords[2, :], 1e-5, 1e5)
            cur_coords[:2, :] /= cur_coords[2:3, :]     # (3, num_points)

            # do image aug
            cur_coords = cur_img_aug_matrix[:3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:3, 3].reshape(3, 1)   # (3, num_points)
            cur_coords = cur_coords[:2, :].transpose(0, 1)  # (num_points,2| u,v)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]    # (x,y)->(y,x)
            # print(cur_coords[0])  --->  [ 39.8298, 893.6777]
            # filter points outside of images
            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )       # (num_points,)
            # for c in range(on_img.shape[0]):
            #     masked_coords = cur_coords[c, on_img[c]].long()
            #     masked_dist = dist[c, on_img[c]]
            #     depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
            masked_coords = cur_coords[on_img].long()       # (num_points_on_img, 2)

            masked_dist = dist[on_img]
            depth[b,0, masked_coords[:,0], masked_coords[:, 1]] = masked_dist
            # (B, 1, H, W)

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots, camera2lidar_trans, intrins, post_rots, 
            post_trans, extra_rots=extra_rots, extra_trans=extra_trans,
        )

        # use points depth to assist the depth prediction in images
        x = self.get_cam_feats(img, depth)      # (B, D, fH, fW, 80)
        x = self.bev_pool(geom, x)              # (B, C, H, W)
        x = self.downsample(x)

        # convert bev features from (b, c, x, y) to (b, c, y, x)
        x = x.permute(0, 1, 3, 2)
        batch_dict['spatial_features_img'] = x
        return batch_dict