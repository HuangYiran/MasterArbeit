import numpy as np
import torch
import os

class cat:
    def forward(self, d1, d2, output = None, clean = False):
        # read data
        tmp1 = np.load(d1)
        tmp2 = np.load(d2)
        # assert
        assert(tmp1.shape[1:] == tmp2.shape[1:])
        # transform to torch.Tensor
        tmp1 = torch.from_numpy(tmp1)
        tmp2 = torch.from_numpy(tmp2)
        # cat
        out = torch.cat([tmp1, tmp2], 0)
        # transfrom back to numpy
        out = out.numpy()
        # save the result 
        if output is not None:
            np.save(output, out)
            if clean:
                os.system('rm '+d1)
                os.system('rm '+d2)
        else:
            np.save(d2, out)
            if clean:
                os.system('rm '+d1)

