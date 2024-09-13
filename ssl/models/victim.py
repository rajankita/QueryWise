import torch.nn as nn


class Victim(nn.Module):
    """class for victim model

    Args:
        nn (_type_): _description_
    """
    def __init__(self, model, input_transform):
        super(Victim, self).__init__()
        self.model = model
        self.input_transform = input_transform
        
    def forward(self, x):
        # change forward here
        x = self.input_transform(x)
        out = self.model(x)
        return out
