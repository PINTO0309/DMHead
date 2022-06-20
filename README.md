# DMHead
Dual model head pose estimation.

## 1. Summary
![icon_design drawio (10)](https://user-images.githubusercontent.com/33194443/174505343-43a2c78c-c86a-4e26-810d-b1cf90965a9d.png)

## 2. Atmosphere


## 3. Model Structure
- INPUTS: `Float32 [N,3,224,224]`
- OUTPUTS: `Float32 [N,3]`, `[Yaw,Roll,Pitch]`

![pinheadpose_1x3x224x224 onnx](https://user-images.githubusercontent.com/33194443/174504855-bf03e294-c9c9-477d-9faf-07b3d0393463.png)
