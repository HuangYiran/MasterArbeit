import torch
import numpy as np

def main():
    a = torch.FloatTensor([[[1,1],[1,2]],[[1,2],[1,1]]])
    b = torch.FloatTensor([[[1,1],[1,2]],[[1,1],[1,2]]])
    out = _cal_cos_distance_pairwise(a,b)
    print out


def _cal_cos_distance_pairwise(hyp, ref):
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

if __name__ == "__main__":
    main()
