<div align="center">

# Efficient Multi-Modal 3D Object Detector via Instance Level Contrastive Distillation

</div>

---

This is the official codes for ["Efficient Multi-Modal 3D Object Detector via Instance Level Contrastive Distillation"](https://arxiv.org/abs/2503.12914), built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [Voxel RCNN](https://github.com/djiajunustc/Voxel-R-CNN) and [BEVFusion](https://github.com/mit-han-lab/bevfusion).

## Framework:
<p align="center">
  <img src="docs/framework1.png" width="95%" height="320">
</p>
<p align="center"><b>Framework:</b> The overview of our proposed method which consists of Instance-level Contrastive Distillation (ICD) and Cross Linear Attention Fusion Module (CLFM)</p>

---

## News:
- [2025-03] Training codes released.
- [2025-02] Submitted IROS 2025.

### 🔥Highlights
* **Strong Performance :** ICD-Fusion 3D detector achieves **SOTA** performance on KITTI validation set for multi-classes 3D Object detection task with single-use data.
* **Great compatibility :** Our approach is **agnostic** to specific detection heads, making it highly adaptable to both single-view and multi-view multimodal 3D object detection tasks.
* **Inference friendly :** Our method enhances performance without imposing excessive computational overhead, ensuring a **computation-friendly** inference.
---

## Installation
1.  Prepare for the running environment.

    You can  follow the installation steps in [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) or download our [conda environment pack]() for easy deployment:

    ```
    cd /user/anaconda3/envs
    mkdir your_env_name
    tar -xzvf your_env_name.tar.gz -C /user/anaconda3/envs/your_env_name
    ```
    
    **NOTE :** Our released implementation is tested on: Ubuntu 20.04, Python 3.7~3.9 (3.9 recommended), Pytorch 2.0.1, cuda 11.8, RTX 4090 (or 3090, 4070) GPUs.


2. Prepare for the data.

* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
    ```
    ICDFusion
    ├── data
    │   ├── kitti
    │   │   │── ImageSets
    │   │   │── training
    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
    │   │   │── testing
    │   │   │   ├──calib & velodyne & image_2
    ├── pcdet
    ├── tools
    ```
* Generate the data infos by running the following command: 
    ```
    python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
    ```
    **NOTE :** If you put KITTI dataset in another directory, please replace the `ROOT_PATH` in `pcdet.datasets.kitti.kitti_dataset` with your customized path before running the above command. 

    Anyway, the final data structure should be organized as follows:

    ```
    ICDFusion
    ├── data
    │   ├── kitti
    │   │   │── ImageSets
    │   │   │── training
    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & velodyne_reduced
    │   │   │── testing
    │   │   │   ├──calib & velodyne & image_2 & velodyne_reduced
    │   │   │── gt_database
    │   │   │── kitti_dbinfos_train.pkl
    │   │   │── kitti_infos_test.pkl
    │   │   │── kitti_infos_train.pkl
    │   │   │── kitti_infos_trainval.pkl
    │   │   │── kitti_infos_val.pkl
    ├── pcdet
    ├── tools
    ```

3. Setup
    ```
    cd ICDFusion
    python setup.py develop
    ```
---

## Getting Started
0. Preparation

    Following the instructions offered by [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), we pretrained Voxel-RCNN as teacher model in our ICD framework. You can directly download the pretrained weights [here](). The choice of the teacher model remains flexible, as long as the encoder's output maintains the same size with student model. Following [BEVFusion](https://github.com/mit-han-lab/bevfusion), you also need to download [SwinTransformer]() pretrained model as the image encoder in our dual-branch encoder.

1. Training
    ```
    cd tools
    python3 train.py --cfg_file cfgs/kitti_models/ICDfusion.yaml
    ```

2. Evaluation 
    ```
    cd tools
    python3 test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
    ```

3. Model Selection
    
    You can use `select_AP40.py` for optimal model selection. Before the analysis, you should set your path to result logfile written by Tensorboard. The logfile is usually under the path: `/ICDFusion/output`

---

## Acknowledgement

We thank these great works and open-source repositories: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), [BEVFusion](https://github.com/mit-han-lab/bevfusion), [Voxel-RCNN](https://github.com/djiajunustc/Voxel-R-CNN), [VirConv](https://github.com/hailanyi/VirConv), [SFD](https://github.com/LittlePey/SFD), [LoGoNet](https://github.com/PJLab-ADG/LoGoNet)

---

## License
This project is free software made available under the MIT License. For details see the LICENSE file.

---

## Citation
```
@inproceedings{su2025icdfusion,
  title={Efficient Multi-Modal 3D Object Detector via Instance Level Contrastive Distillation},
  author={Zhuoqun Su and Huimin Lu and Shuaifeng Jiao and Junhao Xiao and Yaonan Wang and Xieyuanli Chen},
  journal={arXiv preprint arXiv:2503.12914},
  year={2025},
  url= {https://arxiv.org/abs/2503.12914}
}
```
