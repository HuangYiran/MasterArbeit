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
        if opt.rank:
            with file(opt.src_sys2) as fi:
                self.data_sys2 = torch.from_numpy(numpy.load(fi))
        with file(opt.src_ref) as fi:
            self.data_ref = torch.from_numpy(numpy.load(fi))
        self.data_in = torch.cat((self.data_sys, self.data_ref), 1)
        if opt.rank:
            self.data_in = torch.cat((self.data_sys, self.data_sys2, self.data_ref), 1)
        self.nu_batch = len(self.data_in)/self.batch_size

        self.data_tgt = []
        with open(opt.tgt) as fi:
            for line in fi:
                self.data_tgt.append(float(line.strip()))
        self.data_tgt = torch.Tensor(self.data_tgt)

        # read val data
        with file(opt.src_val_sys) as fi:
            self.data_val_sys = torch.from_numpy(numpy.load(fi))
        if opt.rank:
            with file(opt.src_val_sys2) as fi:
                self.data_val_sys2 = torch.from_numpy(numpy.load(fi))
        with file(opt.src_val_ref) as fi:
            self.data_val_ref = torch.from_numpy(numpy.load(fi))
        self.data_val_in = torch.cat((self.data_val_sys, self.data_val_ref), 1)
        if opt.rank:
            self.data_in = torch.cat((self.data_val_sys, self.data_val_sys2, self.data_ref), 1)
        self.nu_val_batch = len(self.data_val_in)/self.batch_size
        self.data_val_tgt = []
        with open(opt.tgt_val) as fi:
            for line in fi:
                self.data_val_tgt.append(float(line.strip()))
        self.data_val_tgt = torch.Tensor(self.data_val_tgt)
        
        # read test data
        with file(opt.src_test_sys) as fi:
            self.data_test_sys = torch.from_numpy(numpy.load(fi))
        if opt.rank:
            with file(opt.src_test_sys2) as fi:
                self.data_test_sys2 = torch.from_numpy(numpy.load(fi))
        with file(opt.src_test_ref) as fi:
            self.data_test_ref = torch.from_numpy(numpy.load(fi))
        self.data_test_in = torch.cat((self.data_test_sys, self.data_test_ref), 1)
        if opt.rank:
            self.data_test_in = torch.cat((self.data_test_sys, self.data_test_sys1, self.data_test_ref), 1)
        self.nu_test_batch = len(self.data_test_in)/self.batch_size
        self.data_test_tgt = []
        with open(opt.tgt_test) as fi:
            for line in fi:
                self.data_test_tgt.append(float(line.strip()))
        self.data_test_tgt = torch.Tensor(self.data_test_tgt)
        
        # shuffle
        self._shuffle2()

    def _shuffle(self):
        """
        !!! Aborded
        only shuffle the training data for last hidden value 
        """
        print("shuffling the dataset")
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
        
    def _shuffle2(self):
        """
        implement shuffle function with torch.index_select(), can be used for both last hidden and full hidden
        the input data and target data here should be type of torch.Tensor or Variable
        """
        random.seed(34843)
        order = range(len(self.data_in))
        random.shuffle(order)
        order = torch.LongTensor(order)
        self.data_in = self.data_in.index_select(0, order)
        self.data_tgt = self.data_tgt.index_select(0, order)
        
        # the same operate for the val and test dataset
        order = range(len(self.data_val_in))
        random.shuffle(order)
        order = torch.LongTensor(order)
        self.data_val_in = self.data_val_in.index_select(0, order)
        self.data_val_tgt = self.data_val_tgt.index_select(0, order)

        order = range(len(self.data_test_in))
        random.shuffle(order)
        order = torch.LongTensor(order)
        self.data_test_in = self.data_test_in.index_select(0, order)
        self.data_test_tgt = self.data_test_tgt.index_select(0, order)

    def normalize_z_score2(self):
        """
        Aborded
        wrong example
        only for last hidden 
        normalize the data with z-score
        Problem: should i change the original data other simplily return teh normalized data???
        """
        data_in_numpy = self.data_in.numpy()
        mean = numpy.mean(data_in_numpy)
        std = numpy.std(data_in_numpy)
        self.data_in = (data_in_numpy - mean)/std
        self.data_in = torch.from_numpy(self.data_in)

        data_val_in_numpy = self.data_val_in.numpy()
        mean = numpy.mean(data_val_in_numpy)
        std = numpy.std(data_val_in_numpy)
        self.data_val_in = (data_val_in_numpy - mean)/std
        self.data_val_in = torch.from_numpy(self.data_val_in)

        data_test_in_numpy = self.data_test_in.numpy()
        mean = numpy.mean(data_test_in_numpy)
        std = numpy.std(data_test_in_numpy)
        self.data_test_in = (data_test_in_numpy - mean)/std
        self.data_test_in = torch.from_numpy(self.data_test_in)
    
    def normalize_z_score(self):
        self.data_in = self._normalize_z_score(self.data_in)
        self.data_val_in = self._normalize_z_score(self.data_val_in)
        self.data_test_in = self._normalize_z_score(self.data_test_in)
    
    def _normalize_z_score(self, data_in):
        data_in_size = data_in.size()
        data_in = data_in.view(-1, data_in_size[-1])
        data_out = []
        for index in range(data_in_size[-1]):
            data_slice = data_in[:, index]
            mean = torch.mean(data_slice)
            std = torch.std(data_slice)
            data_tmp = (data_slice - mean)/std
            data_out.append(data_tmp)
        data_out = torch.stack(data_out, dim = 0)
        data_out = data_out.view(data_in_size)
        return data_out

    
    def normalize_minmax2(self, new_min = -1, new_max = 1):
        """
        wrong example
        only for last hidden value
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
    
    def normalize_minmax(self, new_min = -1, new_max = 1):
        """
        x' = (x - min)/(max - min) * (new_max - new_min) + new_min
        """
        self.data_in = self._normalize_minmax(self.data_in, new_min, new_max)
        self.data_val_in = self._normalize_minmax(self.data_val_in, new_min, new_max)
        self.data_test_in = self._normalize_minmax(self.data_test_in, new_min, new_max)
    
    def _normalize_minmax(self, data_in, new_min = -1, new_max = 1):
        """
        input: Tensor of size (batch_size, seq_len, num_dim) 
        output: Tensor of size (batch_size, seq_len, num_dim)
        """
        data_in_size = data_in.size()
        data_in = data_in.view(-1, data_in_size[-1])
        data_out = []
        for index in range(data_in_size[-1]):
            data_slice = data_in[:, index]
            min = data_slice.min()
            max = data_slice.max()
            data_tmp = (data_slice - min) * (new_max - new_min)/ (max - min) + new_min
            data_out.append(data_tmp)
        data_out = torch.stack(data_out, dim = 1)
        data_out = data_out.view(data_in_size)
        return data_out

    def get_batch(self, sep = False, rep = False):
        """
        input: sep
            sep: !!!Aborded. boolean return the separated data order the combinded data
            rep: if true, when the cur_index reach the end, set it to 0
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_index += 1
        if rep:
            if self.cur_index == self.nu_batch:
                self.cur_index = 0
        start = self.batch_size * self.cur_index
        end = self.batch_size + start
        len_data = len(self.data_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        return (self.data_in[start:end, ], 
                    self.data_tgt[start:end,])
    
    def get_test_batch(self, sep = False, rep = False):
        """
        input: sep
            sep: !!!Aborded. boolean return the separated data order the combinded data
            rep: if true, when the cur_index reach the end, set it to 0
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_test_index += 1
        if rep:
            if self.cur_test_index == self.nu_test_batch:
                self.cur_test_index = 0
        start = self.batch_size * self.cur_test_index
        end = self.batch_size + start
        len_data = len(self.data_test_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data
            
        return (self.data_test_in[start:end, ], 
                    self.data_test_tgt[start:end,])

    def get_val_batch(self, sep = False, rep = False):
        """
        input: sep
            sep: !!!Aborded. boolean return the separated data order the combinded data
            rep: if true, when the cur_index reach the end, set it to 0
        output: (data_sys, data_ref), data_tgt  order data_in, data_tgt
            data_sys: (batch_size, )
            data_ref: (batch_size, )
            data_in : (batch_size, )
            data_tgt: (batch_size, )

        """
        self.cur_val_index += 1
        if rep:
            if self.cur_val_index == self.nu_val_batch:
                self.cur_val_index = 0
        #batch_size = 10
        start = self.batch_size * self.cur_val_index
        end = self.batch_size + start
        len_data = len(self.data_val_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        return (self.data_val_in[start:end, ], 
                    self.data_val_tgt[start:end,])
    
    def get_random_batch(self):
        rands_src = []
        rands_tgt = []
        for i in range(self.batch_size):
            rands_src.append(random.randint(1, len(self.data_in)-1))
            rands_tgt.append(random.randint(1, len(self.data_in)-1))
        rands_src = torch.LongTensor(rands_src)
        rands_tgt = torch.LongTensor(rands_tgt)
        return (self.data_in[rands_src], self.data_tgt[rands_tgt])
    
    def get_nu_batch(self):
        """
        return the number of batch 
        """
        return self.nu_batch, self.nu_val_batch, self.nu_test_batch
    
"""
    def get_batch_repeatly(self, sep = False):
        if self.cur_index == self.nu_batch:
            self.cur_index = 0
            self._shuffle()
        return self.get_batch(sep)
"""
