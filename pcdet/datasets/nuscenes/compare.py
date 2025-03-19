def __getitem__(self, index):
    if self._merge_all_iters_to_one_epoch:
        index = index % len(self.infos)

    info = copy.deepcopy(self.infos[index])
    points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

    input_dict = {
        'points': points,
        'frame_id': Path(info['lidar_path']).stem,
        'metadata': {'token': info['token']}
    }

    if 'gt_boxes' in info:
        if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
            mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
        else:
            mask = None

        input_dict.update({
            'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
            'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
        })

    if self.use_camera:
        input_dict = self.load_camera_info(input_dict, info)

    def load_camera_info(self, input_dict, info):
        '''
        Load RGB images from info['cams']. Preprocess the essential arguments.
        Args:
            input_dict
            info: mainly use info['cams']
        '''
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for _, camera_info in info["cams"].items():
            input_dict["image_paths"].append(camera_info["data_path"])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            input_dict["camera2ego"].append(camera2ego)

            # camera to lidar transform
            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)
        return input_dict

    data_dict = self.prepare_data(data_dict=input_dict)

    def crop_image(self, input_dict):
        '''
        对图像数据进行缩放和裁剪处理, process the image size and record the processed images infos
        '''
        W, H = input_dict["image_shape"]
        img = input_dict["image"]
        if self.training == True:
            fH, fW = self.camera_image_config.FINAL_DIM
            resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN

            # choose a zoom factor in range of resize_lim
            resize = np.random.uniform(*resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = newH - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:
            fH, fW = self.camera_image_config.FINAL_DIM
            resize_lim = self.camera_image_config.RESIZE_LIM_TEST
            resize = np.mean(resize_lim)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = newH - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        
        input_dict['img_process_info'] = [resize, crop, False, 0]
        input_dict['camera_img'] = img
        return input_dict
    
    if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
        gt_boxes = data_dict['gt_boxes']
        gt_boxes[np.isnan(gt_boxes)] = 0
        data_dict['gt_boxes'] = gt_boxes

    if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
        data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
    return data_dict