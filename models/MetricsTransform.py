import torch
class MetricsTransform(torch.nn.Module):
    """
    use mlp to do the metrics_transform: 1-8-4-1
    """
    def __init__(self):
        super(MetricsTransform, self).__init__()
        dim2 = 64
        dim3 = 16
        self.layers = torch.nn.Sequential()
        self.layers.add_module('fc1', torch.nn.Linear(1, dim2))
        self.layers.add_module('bn1', torch.nn.BatchNorm1d(dim2))
        self.layers.add_module('act1', torch.nn.LeakyReLU())
        self.layers.add_module('fc2', torch.nn.Linear(dim2, dim3))
        self.layers.add_module('bn2', torch.nn.BatchNorm1d(dim3))
        self.layers.add_module('act2', torch.nn.LeakyReLU())
        self.layers.add_module('fc3', torch.nn.Linear(dim3,1))

    def forward(self, input):
        """
        input: (batch_size, 1)
        output: 1
        """
        out = self.layers(input)
        return out
