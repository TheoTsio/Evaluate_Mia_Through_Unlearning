import torchvision.models as models
import torch.nn as nn
import torch


class TargetModel_3b(nn.Module):
    def __init__(self, input_size=49152, output_size=10, pretrained=False):
        super(TargetModel_3b, self).__init__()
        # Load ResNet18 (you can use resnet34, resnet50, etc.)
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Replace the final fully connected layer
        # ResNet18's final layer has 512 input features
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, output_size)
        
    def forward(self, x):
        # If input is flattened, reshape to (batch, 3, 128, 128)
        if len(x.shape) == 2:
            x = x.view(x.size(0), 3, 128, 128)
        return self.resnet(x)