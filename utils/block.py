#-*- coding:UTF-8 -*-
import numpy as np

from tqdm import tqdm

def main():
    #docs = ['decMixture_2015_s1', 'decMixture_2015_s2', 'decMixture_2015_ref']
    docs = ['decMixture_2016_ende_ref', 'decMixture_2016_ende_s1', 'decMixture_2016_ende_s2']
    for doc in tqdm(docs):
        dat = np.load('/tmp/'+doc)
        bs, ls, nd = dat.shape
        for i in range(ls):
            tmp = dat[:,i,:]
            np.save('/tmp/'+doc+'_'+str(i), tmp)

if __name__ == '__main__':
    main()
