import torch
import torch.nn as nn

class CNNSmall(nn.Module):
    def __init__(self):
        super(CNNSmall, self).__init__()
        # Note: nn.Sequential inherits nn.Module, which has an attribute self._modules and it is a OrderedDict().
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # (N, 1, 28, 28) -> (N, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (N, 32, 28, 28) -> (N, 32, 14, 14)

            nn.Conv2d(32, 32, kernel_size=3, padding=1),    # (N, 32, 14, 14) -> (N, 32, 14, 14)
            nn.ReLU())
        self.last_pool = nn.MaxPool2d(2, stride=2)          # (N, 32, 14, 14) -> (N, 32, 7, 7)

        self.fc_model = nn.Sequential(
            nn.Linear(1568, 120),
            nn.Tanh(),
            nn.Linear(120, 10))

        # placeholder for the activation map gradients
        self.grad_act = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.grad_act = grad

    def forward(self, x):
        x = self.features(x)

        # register the hook
        x.register_hook(self.activations_hook)

        x = self.last_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return x

    # method for the activation extraction
    def get_activations(self, x):
        return self.features(x)

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.grad_act


