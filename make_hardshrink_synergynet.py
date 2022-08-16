import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x = x[..., np.newaxis]
        yaw = x[:,0,:]
        roll = x[:,1,:]
        pitch = x[:,2,:]
        shrunk_yaw = torch.clip(
            yaw,
            min=torch.tensor(-90.0, dtype=torch.float32),
            max=torch.tensor(90.0, dtype=torch.float32),
        )
        eps = 1e-5
        shrunk_roll = (roll * shrunk_yaw) / (shrunk_yaw + eps)
        shrunk_pitch = (pitch * shrunk_yaw) / (shrunk_yaw + eps)
        output = torch.cat([shrunk_yaw,shrunk_roll,shrunk_pitch], dim=1)
        return output

if __name__ == "__main__":
    model = Model()

    import onnx
    from onnxsim import simplify
    MODEL = f'shrunk_synergynet'
    onnx_file = f"{MODEL}.onnx"

    x = torch.randn(1, 3)

    torch.onnx.export(
        model,
        args=(x),
        f=onnx_file,
        opset_version=11,
        input_names = ['shrunk_input'],
        output_names=['synergynet_shrunk_output'],
    )
    model_onnx1 = onnx.load(onnx_file)
    model_onnx1 = onnx.shape_inference.infer_shapes(model_onnx1)
    onnx.save(model_onnx1, onnx_file)

    model_onnx2 = onnx.load(onnx_file)
    model_simp, check = simplify(model_onnx2)
    onnx.save(model_simp, onnx_file)

    import sys
    sys.exit(0)