import torch
import torch.nn as nn
import torch.nn.functional as nn_functional


class RegressionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)

        self.fc1 = nn.Linear(8 * 8 * 64, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.conv1.weight, mean=0.0, std=0.05, a=-0.1, b=0.1)
        nn.init.constant_(self.conv1.bias, 0.0)

        nn.init.trunc_normal_(self.conv2.weight, mean=0.0, std=0.05, a=-0.1, b=0.1)
        nn.init.constant_(self.conv2.bias, 0.0)

        nn.init.trunc_normal_(self.fc1.weight, mean=0.0, std=0.04, a=-0.08, b=0.08)
        nn.init.constant_(self.fc1.bias, 0.1)

        nn.init.trunc_normal_(self.fc2.weight, mean=0.0, std=0.04, a=-0.08, b=0.08)
        nn.init.constant_(self.fc2.bias, 0.1)

        nn.init.trunc_normal_(self.fc3.weight, mean=0.0, std=1.0 / 192.0, a=-2.0 / 192.0, b=2.0 / 192.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, x):
        # Each original conv_layer is conv -> ReLU -> 2x2 max-pool.
        x = nn_functional.max_pool2d(nn_functional.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = nn_functional.max_pool2d(nn_functional.relu(self.conv2(x)), kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)
        x = nn_functional.relu(self.fc1(x))
        x = nn_functional.relu(self.fc2(x))

        # Output is [x, y], with no sigmoid/clamping in the original model.
        return self.fc3(x)
