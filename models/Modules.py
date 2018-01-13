import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, dim_in, dim_out):
        """
        dim_in: int
        dim_out: int
        """
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.li = nn.Linear(self.dim_in, self.dim_out, bias = True)

    def forward(self, data_in):
        """
        data_in: torch.Variable (batch_size, num_q, num_head * num_dim_v)
        out: torch.Variable (batch_size, num_q, num_dim)
        """
        batch_size, num_q, num_d = data_in.size()
        data_tmp = data_in.view(-1, num_d)
        out = self.li(data_tmp)
        out = out.view(batch_size, num_q, -1)
        return out
