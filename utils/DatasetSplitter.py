# -*- coding: UTF-8 -*-
import torch
import numpy as np
import random
import os

class DatasetSplitter(object):
    def __init__(self, seed = '7777'):
        self.seed = seed

    def split(self, names, shuffle = True):
        """
        input:
            names: list of string
                indicate the dataset(s), the dataset should be type of numpy 
            shuffle: boolean
                weather do the shuffle or not
            syn: boolean
                ob the data in the datalist should stay syn or not
        """
        print("#### Split data ####")
        dataset = {}
        print(">>> Read data")
        for name in names:
            # if type of numpy
            try:
                dataset[name] = np.load(name)
            except IOError:
                dataset[name] = [li.rstrip('\n') for li in open(name)]
        print("<<< finish")
        if shuffle:
            print(">>> Shuffle")
            dataset = self._shuffle(dataset)
            print("<<< finish")
        print(">>> Split")
        self._split_dataset(dataset)
        print("<<< finish")

    def _shuffle(self, dataset):
        """
        implement shuffle function with torch.index_select(), can be used for both last hidden and full hidden
        the input data and target data here should be type of torch.Tensor or Variable
        attention:
            the value of dataset should be type of numpy
        """
        for name, value in dataset.items():
            print('> dataset: '+name)
            print('> seed: '+ str(self.seed))
            print('> length: '+ str(len(value)))
            print('> type of value: ' + str(type(value)))
            random.seed(self.seed)
            order = range(len(value))
            if type(value) == np.ndarray:
                value = torch.from_numpy(value)
                order = torch.LongTensor(order)
                value = value.index_select(0, order)
                dataset[name] = value.numpy()
            else:
                tmp = [value[i] for i in order]
                dataset[name] = tmp
            return dataset

    def _split_dataset(self, dataset, chunk_size = 5000, cut = 30, dim = 1, rm = False):
        """
        split the large dataset into small chunck and save it in the same dir with the suffix '_sub_num'
        input:
            dat: type of dict
                the dataset
            chunk_size:
                the size of each chunck
            cut: None or int
                only get the fisrt 'cut' value in the axis 'dim', possible only with ndarray data
            rm:
                weather remove the origin data or not
        """
        # read data and set attribute
        for name, value in dataset.items():
            print('> dataset: '+ name)
            print('> type of value: '+ str(type(value)))
            if type(value) == np.ndarray:
                # split
                dat = torch.from_numpy(value)
                if len(value.shape) >= dim and cut is not None:
                    index = torch.arange(0, cut).type(torch.LongTensor)
                    dat = torch.index_select(dat, dim, index)
                sp_dat = torch.split(dat, chunk_size)
                # save 
                print('> save data')
                name_out = name + '_sub_'
                for ind, chunk in enumerate(sp_dat):
                    np.save(name_out+str(ind), chunk.numpy())
            else:
                # split
                dat = value
                sp_dat = []
                length = len(value)
                for start in range(0, length, chunk_size):
                    sp_dat.append(dat[start: start+chunk_size])
                # save 
                print('> save data')
                name_out = name + '_sub_'
                for ind, chunk in enumerate(sp_dat):
                    with open(name_out+str(ind), 'w') as fi:
                        for li in chunk:
                            fi.write(li+'\n')
        # clean data
        if rm:
            for ind in dataset.keys():
                os.system('rm '+ ind)


