import torch
import torch.onnx
from facenet_pytorch import InceptionResnetV1
from torch import nn

# Step 1: Load the VGG Face Model (InceptionResnetV1)
model = InceptionResnetV1(pretrained='vggface2').eval()

for p in model.parameters():
    p.requires_grad = False

# Step 3: Test the modified model with a dummy input (to make sure it works)
dummy_input = torch.randn(1, 3, 224, 224)  # Example input, batch size 1, 3 channels (RGB), 160x160 resolution
features = model(dummy_input)

# Print the output feature shape to verify
print(f"Output feature shape: {features.shape}")

# Step 4: Export the model to ONNX format
onnx_model_path = '../../weights/inception_resnet_v1.onnx'
torch.onnx.export(model, dummy_input, onnx_model_path, export_params=True, opset_version=12)

print(f"ONNX model saved to {onnx_model_path}")

