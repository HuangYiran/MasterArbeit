# -*- coding: UTF-8 -*-
import torch
import numpy
import random

class DataUtil(object):
    def __init__(self, opt):
        # read data and shuffle
        self.opt = opt
        self.batch_size = opt.batch_size
        self.cur_index = 0
        self.cur_test_index = 0
        self.cur_val_index = 0

        # read training data
        with file(opt.src_sys) as fi:
            self.data_sys = torch.from_numpy(numpy.load(fi))
        with file(opt.src_ref) as fi:
            self.data_ref = torch.from_numpy(numpy.load(fi))
        self.data_in = torch.cat((self.data_sys, self.data_ref), 1)
        self.nu_batch = len(self.data_in)/self.batch_size

        self.data_tgt = []
        with open(opt.tgt) as fi:
            for line in fi:
                self.data_tgt.append(float(line.strip()))
        self.data_tgt = torch.Tensor(self.data_tgt)
        #self._shuffle()

        # read val data
        with file(opt.src_val_sys) as fi:
            self.data_val_sys = torch.from_numpy(numpy.load(fi))
        with file(opt.src_val_ref) as fi:
            self.data_val_ref = torch.from_numpy(numpy.load(fi))
        self.data_val_in = torch.cat((self.data_val_sys, self.data_val_ref), 1)
        self.nu_val_batch = len(self.data_val_in)/self.batch_size
        self.data_val_tgt = []
        with open(opt.val_tgt) as fi:
            for line in fi:
                self.data_val_tgt.append(float(line.strip()))
        self.data_val_tgt = torch.Tensor(self.data_val_tgt)
        
        # read test data
        with file(opt.src_test_sys) as fi:
            self.data_test_sys = torch.from_numpy(numpy.load(fi))
        with file(opt.src_test_ref) as fi:
            self.data_test_ref = torch.from_numpy(numpy.load(fi))
        self.data_test_in = torch.cat((self.data_test_sys, self.data_test_ref), 1)
        self.nu_test_batch = len(self.data_test_in)/self.batch_size
        self.data_test_tgt = []
        with open(opt.test_tgt) as fi:
            for line in fi:
                self.data_test_tgt.append(float(line.strip()))
        self.data_test_tgt = torch.Tensor(self.data_test_tgt)

    def _shuffle(self):
        """
        最后两行可能有问题，还没有测试
        """
        random.seed(34843)
        order = range(len(self.data_in))
        random.shuffle(order)
        comb = zip(self.data_in, self.data_tgt, order)
        data_in, data_tgt, order = zip(*sorted(comb, key = lambda x: x[-1]))
        self.data_tgt = torch.Tensor(data_tgt)
        tmp_data_in = []
        for i in range(len(data_in)):
            tmp_data_in.append(data_in[i].view(1, -1))
        self.data_in = torch.cat(tmp_data_in, 0)
        #self.data_sys = self.data_in[,1:len(self.data_sys[0])]
        #self.data_ref = self.data_in[,len(self.data_sys[0]):]

    def normalize_z_score(self):
        """
        only training data
        normalize the data with z-score
        Problem: should i change the original data other simplily return teh normalized data???
        """
        data_in_numpy = self.data_in.numpy()
        mean = numpy.mean(data_in_numpy)
        std = numpy.std(data_in_numpy)
        self.data_in = (data_in_numpy - mean)/std
        self.data_in = torch.from_numpy(self.data_in)
    
    def normalize_minmax(self, new_min = -1, new_max = 1):
        """
        normalize the data with mix max, here set the new min and max to -1, 1.
        x' = (x - min)/(max - min) * (new_max - new_min) + new_min
        """
        data_in_numpy = self.data_in.numpy()
        min = numpy.min(data_in_numpy)
        max = numpy.max(data_in_numpy)
        self.data_in = (data_in_numpy - min)*(new_max - new_min)/(max - min) + new_min 
        self.data_in = torch.from_numpy(self.data_in)
        
        data_in_numpy = self.data_val_in.numpy()
        min = numpy.min(data_in_numpy)
        max = numpy.max(data_in_numpy)
        self.data_val_in = (data_in_numpy - min)*(new_max - new_min)/(max - min) + new_min 
        self.data_val_in = torch.from_numpy(self.data_val_in)
        
        data_in_numpy = self.data_test_in.numpy()
        min = numpy.min(data_in_numpy)
        max = numpy.max(data_in_numpy)
        self.data_test_in = (data_in_numpy - min)*(new_max - new_min)/(max - min) + new_min 
        self.data_test_in = torch.from_numpy(self.data_test_in)

    def get_batch(self, sep = False):
        """
        input: sep
            sep: boolean return the separated data order the combinded data
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_index += 1
        start = self.batch_size * self.cur_index
        end = self.batch_size + start + 1
        len_data = len(self.data_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        if sep:
            return ((self.data_sys[start:end,], self.data_ref[start:end,]), 
                    self.data_tgt[start:end,])
        else:
            return (self.data_in[start:end, ], 
                    self.data_tgt[start:end,])
    
    def get_test_batch(self, sep = False):
        """
        input: sep
            sep: boolean return the separated data order the combinded data
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_test_index += 1
        start = self.batch_size * self.cur_test_index
        end = self.batch_size + start + 1
        len_data = len(self.data_test_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        if sep:
            return ((self.data_test_sys[start:end,], self.data_test_ref[start:end,]), 
                    self.data_test_tgt[start:end,])
        else:
            return (self.data_test_in[start:end, ], 
                    self.data_test_tgt[start:end,])

    def get_val_batch(self, sep = False):
        """
        input: sep
            sep: boolean return the separated data order the combinded data
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_val_index += 1
        if self.cur_val_index == self.nu_val_batch:
            self.cur_val_index = 0
        #batch_size = 10
        start = self.batch_size * self.cur_val_index
        end = self.batch_size + start + 1
        len_data = len(self.data_val_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        if sep:
            return ((self.data_val_sys[start:end,], self.data_val_ref[start:end,]), 
                    self.data_val_tgt[start:end,])
        else:
            return (self.data_val_in[start:end, ], 
                    self.data_val_tgt[start:end,])
    
    def get_batch_repeatly(self, sep = False):
        if self.cur_index == self.nu_batch:
            self.cur_index = 0
            self._shuffle()
        return self.get_batch(sep)

    def get_nu_batch(self):
        """
        return the number of batch 
        """
        return self.nu_batch, self.nu_val_batch, self.nu_test_batch
