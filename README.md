# DMHead
Dual model head pose estimation. Fusion of SOTA models. 360° 6D HeadPose detection.

## 1. Summary
![icon_design drawio (12)](https://user-images.githubusercontent.com/33194443/174622614-bc67a307-88df-4577-a1da-daa37c57eeb9.png)

## 2. Atmosphere
https://user-images.githubusercontent.com/33194443/174620267-73c1d26f-796f-40c7-a751-41297b501e77.mp4

## 3. Benchmark
- Trained on 300W-LP (Custom, Mask-wearing face image augmentation)
- Test on AFLW2000
  - June 20, 2022
    ```
    Yaw: 3.6129, Pitch: 5.5801, Roll: 3.8468, MAE: 4.3466
    ```

## 4. Model Structure
- INPUTS: `Float32 [N,3,224,224]`
- OUTPUTS: `Float32 [N,3]`, `[Yaw,Roll,Pitch]`

![pinheadpose_1x3x224x224 onnx](https://user-images.githubusercontent.com/33194443/174504855-bf03e294-c9c9-477d-9faf-07b3d0393463.png)

## 5. Citation
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
