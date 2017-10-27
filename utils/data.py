# -*- coding: UTF-8 -*-
import torch
import numpy
import random

class DataUtil(object):
    def __init__(self, opt ):
        # read data and shuffle
        self.opt = opt
        self.batch_size = opt.batch_size
        self.cur_index = 0

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
        self._shuffle()

    def _shuffle(self):
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
        self.data_sys = self.data_in[1:len(self.data_sys),]
        self.data_ref = self.data_in[len(self.data_sys):,]


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
        start = self.batch_size * self.cur_index - 1
        end = self.batch_size + start
        len_data = len(self.data_in)
        
        if start > len_data:
            print("the data set is empty")
            return None, None
        elif end > len_data:
            end = len_data

        if sep:
            return ((self.ata_sys[start:end,], self.data_ref[start:end,]), 
                    self.data_tgt[start:end,])
        else:
            return (self.data_in[start:end, ], 
                    self.data_tgt[start:end,])

    def get_batch_repeatly(self, sep = False):
        if self.cur_index == self.nu_batch:
            self.cur_index = 0
            self._shuffle()
        return self.get_batch(sep)

    def get_nu_batch(self):
        """
        return the number of batch 
        """
        return self.nu_batch