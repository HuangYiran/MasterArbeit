import numpy as np
import torch

def cal_cos_distance_pairwise(hyp, ref):
    """
    hyp and ref should be type of torch
    hyp and ref should have some shape, because torch.nn.CosineSimilarity only work with some shape
    shape of (batch_size, sen_len, num_dim)
    """
    # assert
    assert(hyp.shape == ref.shape)
    batch_size, sen_len, num_dim = ref.shape
    tmp = np.arange(sen_len)
    cos = torch.nn.CosineSimilarity(dim = 2)
    out = []
    print('Lenght of sentence: ' + str(sen_len))
    for i in range(sen_len):
        indexes = torch.LongTensor(np.roll(tmp, -1*i))
        tmp_ref = torch.index_select(ref, 1, indexes)
        out.append(cos(hyp, tmp_ref))
    out = torch.stack(out, dim = 2)
    return out

def canberra(hyp, ref):
    """
    hyp and ref should be type of torch and should have the same shape
    input:
        hyp: FloatTensor (batch_size, num_dim)
        ref: FloatTensor (batch_size, num_dim)
    """
    return torch.sum(torch.abs(hyp -ref)/(torch.abs(hyp)+torch.abs(ref)), dim = 1)
