import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision.transforms import Resize
class MobileNetV2Backbone(nn.Module):
    def __init__(self):
        super(MobileNetV2Backbone, self).__init__()
        # Load the pre-trained MobileNetV2 model
        self.resize = Resize((224, 224))

        self.mobilenetv2 = models.mobilenet_v2(pretrained=True)
        in_features = self.mobilenetv2.classifier[1].in_features
        self.mobilenetv2.classifier = nn.Sequential(
            nn.Linear(in_features, 256))

        # Modify the last layer to fit your output size if needed

        # Add a resizing layer to resize input images
         # Adjust the target size as needed

    def forward(self, x):
        # Resize the input to match MobileNetV2's required size
        x = self.resize(x)
        x = x.repeat(1, 3, 1, 1)  # Repeat grayscale image to have 3 channels
        # Perform any additional processing if required
        return self.mobilenetv2(x)

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        # Adjust the input dimension to match the output size from the backbone
        self.fc1 = nn.Linear(256, 128)  # Assuming MobileNetV2 output size is 1280
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x

class StarDetector(nn.Module):
    def __init__(self):
        super(StarDetector, self).__init__()
        self.backbone = MobileNetV2Backbone()
        self.Regressor = Regressor()

    def forward(self, x):
        x = self.backbone(x)
        return self.Regressor(x)