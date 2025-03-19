from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points
        return data_dict

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates   
        data_dict['voxel_num_points'] = num_points

        return data_dict
 
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict['images'] = [compose(img) for img in data_dict['images']]
        return data_dict

    def image_crop(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_crop, config=config)
        H, W = data_dict["image_shape"]
        img = data_dict["images"]
        if self.training:
            fH, fW = config.FINAL_DIM
            crop_h = H - fH
            crop_w = np.random.randint(0, max(0, W - fW))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            fH, fW = config.FINAL_DIM
            crop_h = H - fH
            crop_w = (W - fW) // 2
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        img = img[crop_h:crop_h + fH, crop_w:crop_w + fW]
        img_process_info = [1.0, crop, False, 0]
        data_dict['img_process_infos'] = img_process_info
        data_dict['images'] = img
        return data_dict

    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_info = data_dict['img_process_infos']
        resize, crop, flip, rotate = img_process_info

        rotation = torch.eye(2)
        translation = torch.zeros(2)
        # post-homography transformation
        rotation *= resize
        translation -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
        theta = rotate / 180 * np.pi
        A = torch.Tensor(
            [
                [np.cos(theta), np.sin(theta)],
                [-np.sin(theta), np.cos(theta)],
            ]
        )
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        rotation = A.matmul(rotation)
        translation = A.matmul(translation) + b
        transform = torch.eye(4)
        transform[:2, :2] = rotation
        transform[:2, 3] = translation
        data_dict["img_aug_matrix"] = transform
        return data_dict

    def gt_sampling_trans(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.gt_sampling_trans, config=config)
        points = data_dict['points']
        lidar2image = data_dict['lidar2image']  # (3, 4)
        image = data_dict['images']  # (H, W, 3)
        image_shape = data_dict['image_shape']  # (H, W)

        lidar_coords = points[:, :3]  # (N, 3),
        intensities = points[:, 3]  # (N,),

        lidar_coords_h = np.hstack((lidar_coords, np.ones((lidar_coords.shape[0], 1))))  # (N, 4)

        lidar_image_coords = lidar2image @ lidar_coords_h.T  # (3, N)
        lidar_image_coords[:2, :] /= lidar_image_coords[2, :]
        projected_coords = lidar_image_coords[:2, :].T  # (N, 2)

        depths = lidar_image_coords[2, :]  # (N,)

        valid_mask = (
            (projected_coords[:, 0] >= 0) & (projected_coords[:, 0] < image_shape[1]) &
            (projected_coords[:, 1] >= 0) & (projected_coords[:, 1] < image_shape[0]) &
            (depths > 0)
        )
        projected_coords = projected_coords[valid_mask].astype(np.int32)
        depths = depths[valid_mask]
        intensities = intensities[valid_mask]
        lidar_coords = lidar_coords[valid_mask]

        projection_image = np.zeros((image_shape[0], image_shape[1], 5), dtype=np.float32)  # (x, y, z, depth, intensity)

        projection_image[projected_coords[:, 1], projected_coords[:, 0], :3] = lidar_coords  # x, y, z
        projection_image[projected_coords[:, 1], projected_coords[:, 0], 3] = depths    # depth
        projection_image[projected_coords[:, 1], projected_coords[:, 0], 4] = intensities  # intensity

        result_image = np.concatenate((image, projection_image), axis=-1)  # (H, W, 8)
        data_dict['images'] = result_image
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        return data_dict
