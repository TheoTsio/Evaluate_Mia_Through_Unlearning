import torch.nn as nn
import torch

class TargetModel_3c(nn.Module):
    def __init__(self, input_size=600, output_size=100):
        super(TargetModel_3c, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, output_size)
        )

    def forward(self, x):
        # Ensure input is flat (batch_size, 600)
        return self.layers(x)