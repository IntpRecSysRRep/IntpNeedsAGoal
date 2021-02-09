import torch
import torch.nn as nn
from torchvision.models import vgg19

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # get the pretrained VGG19 network
        self.vgg = vgg19(pretrained=True)

        # disect the network to access its last convolutional layer
        #self.features_conv = self.vgg.features[:36]
        self.features_conv = self.vgg.features[:35]
        self.softplus = nn.Softplus()

        # get the max pool of the features stem
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        # get the classifier of the vgg19
        self.classifier = self.vgg.classifier

        # placeholder for the gradients
        self.grad_act = None

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.grad_act = grad

    def forward(self, x):
        x = self.features_conv(x)
        x = self.softplus(x)

        # register the hook
        x.register_hook(self.activations_hook)

        # apply the remaining pooling
        x = self.max_pool(x)
        x = x.view((1, -1))
        x = self.classifier(x)
        return x

    # method for the activation exctraction
    def get_activations(self, x):
        # return self.features_conv(x)
        return self.softplus(self.features_conv(x))

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.grad_act
