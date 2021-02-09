import torch

class BertGradientInterpreter():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        # first_layer = list(self.model.bert.encoder.layer[0]._modules.items())[0][1]     # [0][0] is the string
        first_layer = self.model.bert.encoder.layer[0]
        print(first_layer)
        first_layer.register_backward_hook(hook_function)

    def generate_gradients(self, indexed_tokens, segments_ids, input_masks, labels):
        # Forward
        model_output = self.model(indexed_tokens, segments_ids, input_masks, labels=None)
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(model_output)
        one_hot_output[torch.arange(16), labels] = 1
        model_output.backward(gradient=one_hot_output)
        for grad in self.gradients:
            print(grad.shape)

        return self.gradients.data.numpy()

