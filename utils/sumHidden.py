import torch
import numpy as np
from DatasetSplitter import DatasetSplitter
def main():
    names = ['/tmp/de.en2015decoder_hidden.ref_sub_0.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_1.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_2.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_3.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_4.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_5.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_6.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_7.npy',
            '/tmp/de.en2015decoder_hidden.ref_sub_8.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_0.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_1.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_2.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_3.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_4.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_5.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_6.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_7.npy',
            '/tmp/de.en2015decoder_hidden.s1_sub_8.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_0.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_1.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_2.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_3.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_4.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_5.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_6.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_7.npy',
            '/tmp/de.en2015decoder_hidden.s2_sub_8.npy',
            '/tmp/de.en2016decoder_hidden.ref_sub_0.npy',
            '/tmp/de.en2016decoder_hidden.ref_sub_1.npy',
            '/tmp/de.en2016decoder_hidden.ref_sub_2.npy',
            '/tmp/de.en2016decoder_hidden.ref_sub_3.npy',
            '/tmp/de.en2016decoder_hidden.ref_sub_4.npy',
            '/tmp/de.en2016decoder_hidden.s1_sub_0.npy',
            '/tmp/de.en2016decoder_hidden.s1_sub_1.npy',
            '/tmp/de.en2016decoder_hidden.s1_sub_2.npy',
            '/tmp/de.en2016decoder_hidden.s1_sub_3.npy',
            '/tmp/de.en2016decoder_hidden.s1_sub_4.npy',
            '/tmp/de.en2016decoder_hidden.s2_sub_0.npy',
            '/tmp/de.en2016decoder_hidden.s2_sub_1.npy',
            '/tmp/de.en2016decoder_hidden.s2_sub_2.npy',
            '/tmp/de.en2016decoder_hidden.s2_sub_3.npy',
            '/tmp/de.en2016decoder_hidden.s2_sub_4.npy',
            ]
    for name in names:
        print('> process: '+ name)
        tmp = np.load(name)
        tmp = tmp.sum(axis = 1)
        np.save(name[:-4], tmp)

if __name__ == '__main__':
    main()
