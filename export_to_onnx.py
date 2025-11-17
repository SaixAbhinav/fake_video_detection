import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

class FakeImageDetector(nn.Module):
    def __init__(self):
        super(FakeImageDetector, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model
model = FakeImageDetector()
model.load_state_dict(torch.load("C:/Users/saiab/Desktop/code/models/fake_image_detector.pth", map_location="cpu"))
model.eval()

# Dummy input (correct size for your model)
dummy_input = torch.randn(1, 3, 128, 128)

# Export to ONNX using the legacy exporter (more stable for standard models)
print("Exporting model to ONNX format...")
torch.onnx.export(
    model,
    dummy_input,
    "fake_image_detector.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=12,
    verbose=False,
    dynamo=False  # Use the legacy exporter for compatibility
)
print("Model exported successfully to fake_image_detector.onnx")

print("Exported to fake_image_detector.onnx")