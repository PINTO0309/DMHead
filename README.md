# DMHead
Dual model head pose estimation. Fusion of SOTA models. 360° 6D HeadPose detection. All pre-processing and post-processing are fused together, allowing end-to-end processing in a single inference.

## 1. Summary
- **`[Front side]`** Wearing a mask mode - 6DRepNet (RepVGG-B1g2)

  ![icon_design drawio (12)](https://user-images.githubusercontent.com/33194443/174622614-bc67a307-88df-4577-a1da-daa37c57eeb9.png)
  ![image](https://user-images.githubusercontent.com/33194443/175760351-bd8d2e61-bb49-48f3-8023-c45c12cbd800.png)

- **`[Front side]`** Not wearing a mask mode - SynergyNet (MobileNetV2)

  ![icon_design drawio (14)](https://user-images.githubusercontent.com/33194443/175760025-b359e1d2-ac16-456e-8cf6-2c58514fbc7c.png)
  ![image](https://user-images.githubusercontent.com/33194443/174690800-272e5a06-c932-414f-8397-861d7d6284d0.png)

- **`[Rear side]`** WHENet

  ![image](https://user-images.githubusercontent.com/33194443/175760218-4e61da30-71b6-4d2a-8ca4-ddc4c2ec5df0.png)

## 2. Inference Test

```bash
wget https://github.com/PINTO0309/DMHead/releases/download/1.1.0/yolov4_headdetection_480x640_post.onnx
wget https://github.com/PINTO0309/DMHead/releases/download/1.1.0/dmhead_mask_Nx3x224x224.onnx
wget https://github.com/PINTO0309/DMHead/releases/download/1.1.0/dmhead_nomask_Nx3x224x224.onnx

python demo_video.py
```

## 3. Atmosphere
https://user-images.githubusercontent.com/33194443/174620267-73c1d26f-796f-40c7-a751-41297b501e77.mp4

https://user-images.githubusercontent.com/33194443/175073709-e9c43655-27a9-4760-a38c-768dabe33c1f.mp4

## 4. Benchmark
- Trained on 300W-LP (Custom, Mask-wearing face image augmentation)
- Test on AFLW2000
  - June 20, 2022
    ```
    Yaw: 3.6129, Pitch: 5.5801, Roll: 3.8468, MAE: 4.3466
    ```
  - July 3, 2022
    ```
    Yaw: 3.3417, Pitch: 5.0580, Roll: 3.5549, MAE: 3.9848
    ```

## 5. Model Structure
- INPUTS: `Float32 [N,3,224,224]`
- OUTPUTS: `Float32 [N,3]`, `[Yaw,Roll,Pitch]`

<details><summary>Click to expand</summary><div>

  ![pinheadpose_1x3x224x224 onnx](https://user-images.githubusercontent.com/33194443/174504855-bf03e294-c9c9-477d-9faf-07b3d0393463.png)

</div></details>
  
## 6. References
1. https://github.com/choyingw/SynergyNet
2. https://github.com/thohemp/6DRepNet
3. https://github.com/Ascend-Research/HeadPoseEstimation-WHENet

## 7. Citation
```
@misc{https://doi.org/10.48550/arxiv.2005.10353,
    doi = {10.48550/ARXIV.2005.10353},
    url = {https://arxiv.org/abs/2005.10353},
    author = {Zhou, Yijun and Gregson, James},
    title = {WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose},
    publisher = {arXiv},
    year = {2020},
}
```
```
@misc{hempel20226d,
    title={6D Rotation Representation For Unconstrained Head Pose Estimation},
    author={Thorsten Hempel and Ahmed A. Abdelrahman and Ayoub Al-Hamadi},
    year={2022},
    eprint={2202.12555},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
```
@INPROCEEDINGS{wu2021synergy,
  author={Wu, Cho-Ying and Xu, Qiangeng and Neumann, Ulrich},
  booktitle={2021 International Conference on 3D Vision (3DV)},
  title={Synergy between 3DMM and 3D Landmarks for Accurate 3D Facial Geometry},
  year={2021}
}
```
