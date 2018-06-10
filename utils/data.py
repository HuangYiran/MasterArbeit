# -*- coding: UTF-8 -*-
import torch
import numpy
import random
import os

from sklearn.decomposition import PCA

class DataUtil(object):
    def __init__(self, opt):
        # read data and shuffle
        self.opt = opt
        self.batch_size = opt.batch_size
        self.cur_index = 0
        self.cur_test_index = 0
        self.cur_val_index = 0
        self.is_splitted = opt.is_splitted # if set, read the splitted data instead
        self.num_chunk = 1
        self.num_val_chunk = 1
        self.num_test_chunk = 1
        self.whiten = True # only for 2D input

        # read training data
        if not self.is_splitted:
            if opt.cross_val:
                self._load_data_cv(opt.src_sys, opt.src_sys2, opt.src_ref, opt.tgt, opt.rank, opt.sf_output, opt.cv_val_index)
            else:
                self._load_data(opt.src_sys, opt.src_sys2, opt.src_ref, opt.tgt, opt.src_val_sys, opt.src_val_sys2, opt.src_val_ref, opt.tgt_val, opt.src_test_sys, opt.src_test_sys2, opt.src_test_ref, opt.tgt_test, opt.rank, opt.sf_output)
        else: 
            # is splitted
            self.ind_chunk = 0
            self.ind_val_chunk = 0
            self.ind_test_chunk = 0
            self.num_chunk = self._get_num_chunk(opt.src_sys)
            self.num_val_chunk = self._get_num_chunk(opt.src_val_sys)
            self.num_test_chunk = self._get_num_chunk(opt.src_test_sys)
            #assert
            assert(self._get_num_chunk(opt.src_sys) == self._get_num_chunk(opt.src_sys2) and
                    self._get_num_chunk(opt.src_sys) == self._get_num_chunk(opt.src_ref) and
                    self._get_num_chunk(opt.src_sys) == self._get_num_chunk(opt.tgt))
            assert(self._get_num_chunk(opt.src_val_sys) == self._get_num_chunk(opt.src_val_sys2) and
                    self._get_num_chunk(opt.src_val_sys) == self._get_num_chunk(opt.src_val_ref) and
                    self._get_num_chunk(opt.src_val_sys) == self._get_num_chunk(opt.tgt_val))
            assert(self._get_num_chunk(opt.src_test_sys) == self._get_num_chunk(opt.src_test_sys2) and
                    self._get_num_chunk(opt.src_test_sys) == self._get_num_chunk(opt.src_test_ref) and
                    self._get_num_chunk(opt.src_test_sys) == self._get_num_chunk(opt.tgt_test))
            # read first chunk
            suffix = '_sub_'+str(self.ind_chunk)+'.npy'
            if opt.cross_val:
                self._load_data_cv(opt.src_sys+suffix, opt.src_sys2+suffix, opt.src_ref+suffix, opt.tgt+suffix[:-4], opt.rank, opt.sf_output, opt.cv_val_index+suffix)
            else:
                self._load_data(opt.src_sys+suffix, opt.src_sys2+suffix, opt.src_ref+suffix, opt.tgt+suffix[:-4], opt.src_val_sys+suffix, opt.src_val_sys2+suffix, opt.src_val_ref+suffix, opt.tgt_val+suffix[:-4], opt.src_test_sys+suffix, opt.src_test_sys2+suffix, opt.src_test_ref+suffix, opt.tgt_test+suffix[:-4], opt.rank, opt.sf_output)

        # shuffle
        self._shuffle2()
    
    def _get_num_chunk(self, dat):
        """
        check how many files are there with the name of 'dat'+_sub_num
        """
        dir_dat = os.path.dirname(dat)
        dir_basename = os.path.basename(dat)
        li_files = os.listdir(dir_dat)
        counter = 0
        for i in li_files:
            if dir_basename+'_sub_' in i:
                counter+=1
        return counter

    def _is_numeric(self, x):
        return x in ['-', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'e']
    
    def _load_data(self, src_sys, src_sys2, src_ref, tgt, src_val_sys, src_val_sys2, src_val_ref, tgt_val, src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output):
        # read training data
        self._load_training_data(src_sys, src_sys2, src_ref, tgt, rank, sf_output)
        # read val data
        self._load_validating_data(src_val_sys, src_val_sys2, src_val_ref, tgt_val, rank, sf_output)
        # read test data
        self._load_testing_data(src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output)
        # assert
        assert(len(self.data_tgt) == len(self.data_in))
        assert(len(self.data_val_tgt) == len(self.data_val_in))
        assert(len(self.data_test_tgt)==len(self.data_test_in))

    def _load_training_data(self, src_sys, src_sys2, src_ref, tgt, rank, sf_output):
        # read training data
        with file(src_sys) as fi:
            self.data_sys = numpy.load(fi)
            if self.whiten:
                self.data_sys = self._pca_whiten(self.data_sys)
            self.data_sys = torch.from_numpy(self.data_sys)
        if rank:
            with file(src_sys2) as fi:
                self.data_sys2 = numpy.load(fi)
                if self.whiten:
                    self.data_sys2 = self._pca_whiten(self.data_sys2)
                self.data_sys2 = torch.from_numpy(self.data_sys2)
        with file(src_ref) as fi:
            self.data_ref = numpy.load(fi)
            if self.whiten:
                self.data_ref = self._pca_whiten(self.data_ref)
            self.data_ref = torch.from_numpy(self.data_ref)
        self.data_in = torch.cat((self.data_sys, self.data_ref), 1)
        if rank:
            self.data_in = torch.cat((self.data_sys, self.data_sys2, self.data_ref), 1)
        self.nu_batch = len(self.data_in)/self.batch_size

        self.data_tgt = []
        with open(tgt) as fi:
            for line in fi:
                if sf_output:
                    self.data_tgt.append(int(line.strip()))
                else:
                    tmp = filter(self._is_numeric, line)
                    if len(tmp) == 0:
                        continue
                    try:
                        self.data_tgt.append(float(tmp))
                    except ValueError:
                        print "+++"+tmp+"+++"
        if sf_output:
            #self.data_tgt = list(map(self._result_transform_softmax, self.data_tgt))
            self.data_tgt = torch.LongTensor(self.data_tgt)
            self.data_tgt = self.data_tgt + 1
        else:
            self.data_tgt = torch.FloatTensor(self.data_tgt)

    def _load_validating_data(self, src_val_sys, src_val_sys2, src_val_ref, tgt_val, rank, sf_output):
        # read val data
        with file(src_val_sys) as fi:
            self.data_val_sys = numpy.load(fi)
            if self.whiten:
                self.data_val_sys = self._pca_whiten(self.data_val_sys)
            self.data_val_sys = torch.from_numpy(self.data_val_sys)
        if rank:
            with file(src_val_sys2) as fi:
                self.data_val_sys2 = numpy.load(fi)
                if self.whiten:
                    self.data_val_sys2 = self._pca_whiten(self.data_val_sys2)
                self.data_val_sys2 = torch.from_numpy(self.data_val_sys2)
        with file(src_val_ref) as fi:
            self.data_val_ref = numpy.load(fi)
            if self.whiten:
                self.data_val_ref = self._pca_whiten(self.data_val_ref)
            self.data_val_ref = torch.from_numpy(self.data_val_ref)
        self.data_val_in = torch.cat((self.data_val_sys, self.data_val_ref), 1)
        if rank:
            self.data_val_in = torch.cat((self.data_val_sys, self.data_val_sys2, self.data_val_ref), 1)
        self.nu_val_batch = len(self.data_val_in)/self.batch_size
        
        self.data_val_tgt = []
        with open(tgt_val) as fi:
            for line in fi:
                tmp = filter(self._is_numeric, line)
                if len(tmp) == 0:
                        continue
                if sf_output:
                    self.data_val_tgt.append(int(line.strip()))
                else:
                    self.data_val_tgt.append(float(tmp))
        if sf_output:
            #self.data_val_tgt = list(map(self._result_transform_softmax, self.data_val_tgt))
            self.data_val_tgt = torch.LongTensor(self.data_val_tgt)
            self.data_val_tgt = self.data_val_tgt + 1
        else:
            self.data_val_tgt = torch.FloatTensor(self.data_val_tgt)
 
    def _load_testing_data(self, src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output):
        # read test data
        with file(src_test_sys) as fi:
            self.data_test_sys = numpy.load(fi)
            if self.whiten:
                self.data_test_sys = self._pca_whiten(self.data_test_sys)
            self.data_test_sys = torch.from_numpy(self.data_test_sys)
        if rank:
            with file(src_test_sys2) as fi:
                self.data_test_sys2 = numpy.load(fi)
                if self.whiten:
                    self.data_test_sys2 = self._pca_whiten(self.data_test_sys2)
                self.data_test_sys2 = torch.from_numpy(self.data_test_sys2)
        with file(src_test_ref) as fi:
            self.data_test_ref = numpy.load(fi)
            if self.whiten:
                self.data_test_ref = self._pca_whiten(self.data_test_ref)
            self.data_test_ref = torch.from_numpy(self.data_test_ref)
        self.data_test_in = torch.cat((self.data_test_sys, self.data_test_ref), 1)
        if rank:
            self.data_test_in = torch.cat((self.data_test_sys, self.data_test_sys2, self.data_test_ref), 1)
        self.nu_test_batch = len(self.data_test_in)/self.batch_size
        
        self.data_test_tgt = []
        with open(tgt_test) as fi:
            for line in fi:
                tmp = filter(self._is_numeric, line)
                if len(tmp) == 0:
                        continue
                if sf_output:
                    self.data_test_tgt.append(int(line.strip()))
                else:
                    self.data_test_tgt.append(float(tmp))
        if sf_output:
            #self.data_test_tgt = list(map(self._result_transform_softmax, self.data_test_tgt))
            self.data_test_tgt = torch.LongTensor(self.data_test_tgt)
            self.data_test_tgt = self.data_test_tgt + 1
        else:
            self.data_test_tgt = torch.FloatTensor(self.data_test_tgt)

    def _load_data_cv(self, src_sys, src_sys2, src_ref, src_tgt, rank, sf_output, cv_val_index):
        """
        only use the data store in the src docs. distribute the data into 10 chunks. The 'cv_val_index' chunk is the valiation data
        """
        # read data
        tmp_sys = torch.from_numpy(numpy.load(src_sys))
        if rank:
            tmp_sys2 = torch.from_numpy(numpy.load(src_sys2))
        tmp_ref = torch.from_numpy(numpy.load(src_ref))
        tmp_in = torch.cat((tmp_sys, tmp_ref), 1)
        if rank:
            tmp_in = torch.cat((tmp_sys, tmp_sys2, tmp_ref), 1)
        tmp_tgt = []
        with open(src_tgt) as fi:
            for line in fi:
                tmp = filter(self._is_numeric, line)
                if len(tmp) == 0:
                    continue
                if sf_output:
                    tmp_tgt.append(int(tmp))
                else:
                    tmp_tgt.append(float(tmp))
        if sf_output:
            tmp_tgt = torch.LongTensor(tmp_tgt)
            tmp_tgt = tmp_tgt + 1
        else:
            tmp_tgt = torch.FloatTensor(tmp_tgt)
        assert(len(tmp_tgt) == len(tmp_in))
        # shuffle
#        len_tmp_in = len(tmp_in)
#        order = range(len_tmp_in)
#        random.seed(24324)
#        random.shuffle(order)
#        order = torch.LongTensor(order)
#        tmp_in = tmp_in.index_select(0, order)
#        tmp_tgt = tmp_tgt.index_select(0, order)
        # distribute the data 
        len_tmp_in = len(tmp_in)
        val_begin = int(0.1*cv_val_index*len_tmp_in)
        val_end = int(0.1*(cv_val_index+1)*len_tmp_in)
        val_indexes = range(val_begin, val_end)
        val_indexes = torch.LongTensor(val_indexes)
        train_indexes = range(0, val_begin)
        train_indexes2 = range(val_end, len_tmp_in)
        train_indexes.extend(train_indexes2)
        train_indexes = torch.LongTensor(train_indexes)
        # extrac data
        self.data_in = tmp_in.index_select(0, train_indexes)
        self.data_tgt = tmp_tgt.index_select(0, train_indexes)
        self.data_val_in = tmp_in.index_select(0, train_indexes)
        self.data_val_tgt = tmp_tgt.index_select(0, train_indexes)
        self.data_test_in = tmp_in.index_select(0, val_indexes)
        self.data_test_tgt = tmp_tgt.index_select(0, val_indexes)
        self.nu_batch = len(self.data_in)/self.batch_size
        self.nu_val_batch =  len(self.data_val_in)/self.batch_size
        self.nu_test_batch = len(self.data_test_in)/self.batch_size

    def _pca_whiten(self, data, n_components = 300, svd_solver = 'auto'):
        """
        data set should be 2D and type of numpy
        """
        pca = PCA(n_components, svd_solver = svd_solver)
        pca.fit(data)
        return pca.transform(data).astype(numpy.float32)

    def _result_transform_softmax(self, x):
        if x == -1.0:
            return [0, 0, 1]
        elif x == 0.0:
            return [0, 1, 0]
        elif x == 1.0:
            return [1, 0, 0]
        else:
            raise("unknown result error")

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
        print order.shape
        print self.data_test_in.shape
        print self.data_test_tgt.shape
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
        if self.is_splitted:
            # check the index of the chunk first
            if self.cur_index == self.nu_batch and self.ind_chunk < self.num_chunk-1:
                self.cur_index = 0
                self.ind_chunk += 1
                suffix = '_sub_'+str(self.ind_chunk)+'.npy'
                self._load_training_data(self.opt.src_sys+suffix, self.opt.src_sys2+suffix, self.opt.src_ref+suffix, self.opt.tgt+suffix[:-4], self.opt.rank, self.opt.sf_output)
                self._shuffle2()
            if rep and self.ind_chunk == self.num_chunk - 1:
                self.cur_index = 0
                self.ind_chunk = 0
                suffix = '_sub_'+str(self.ind_chunk)+'.npy'
                self._load_training_data(self.opt.src_sys+suffix, self.opt.src_sys2+suffix, self.opt.src_ref+suffix, self.opt.tgt+suffix[:-4], self.opt.rank, self.opt.sf_output)
                self._shuffle2()
        elif rep:
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
        self.cur_index += 1
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
        if self.is_splitted:
            # check the index of the chunk first
            if self.cur_test_index == self.nu_test_batch and self.ind_test_chunk < self.num_test_chunk-1:
                self.cur_test_index = 0
                self.ind_test_chunk += 1
                suffix = '_sub_'+str(self.ind_test_chunk) + '.npy'
                self._load_testing_data(self.opt.src_test_sys+suffix, self.opt.src_test_sys2+suffix, self.opt.src_test_ref+suffix, self.opt.tgt_test+suffix[:-4], self.opt.rank, self.opt.sf_output)
            if rep and self.ind_test_chunk == self.num_test_chunk - 1:
                self.cur_test_index = 0
                self.ind_test_chunk = 0
                suffix = '_sub_'+str(self.ind_test_chunk) + '.npy'
                self._load_testing_data(self.opt.src_test_sys+suffix, self.opt.src_test_sys2+suffix, self.opt.src_test_ref+suffix, self.opt.tgt_test+suffix[:-4], self.opt.rank, self.opt.sf_output)
        elif rep:
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
        self.cur_test_index += 1            
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
        if self.is_splitted:
            # check the index of the chunk first
            if self.cur_val_index == self.nu_val_batch and self.ind_val_chunk < self.num_val_chunk-1:
                self.cur_val_index = 0
                self.ind_val_chunk += 1
                suffix = '_sub_'+str(self.ind_val_chunk) + '.npy'
                self._load_validating_data(self.opt.src_val_sys+suffix, self.opt.src_val_sys2+suffix, self.opt.src_val_ref+suffix, self.opt.tgt_val+suffix[:-4], self.opt.rank, self.opt.sf_output)
            if rep and self.cur_val_index == self.nu_val_batch and self.ind_val_chunk == self.num_val_chunk - 1:
                self.cur_val_index = 0
                self.ind_val_chunk = 0
                suffix = '_sub_'+str(self.ind_val_chunk) + '.npy'
                self._load_validating_data(self.opt.src_val_sys+suffix, self.opt.src_val_sys2+suffix, self.opt.src_val_ref+suffix, self.opt.tgt_val+suffix[:-4], self.opt.rank, self.opt.sf_output)
        elif rep:
            if self.cur_val_index == self.nu_val_batch:
                self.cur_val_index = 0
        #batch_size = 10
        start = self.batch_size * self.cur_val_index
        end = self.batch_size + start
        len_data = len(self.data_val_in)
        if start > len_data:
            print("the data set is empty")
            print("nu_val_batch: " + str(self.nu_val_batch))
            print("num_val_chunk: " + str(self.num_val_chunk))
            print("cur_val_index: " + str(self.cur_val_index))
            print("ind_val_chunk: " + str(self.ind_val_chunk))
            return None, None
        elif end > len_data:
            end = len_data
        self.cur_val_index += 1

        return (self.data_val_in[start:end, ], 
                    self.data_val_tgt[start:end,])
    
    def reset_cur_val_index(self):
        self.cur_val_index = 0
        self.ind_val_chunk = 0

    def reset_cur_test_index(self):
        self.cur_test_index = 0
        self.ind_test_chunk = 0

    
    def get_random_batch(self):
        """
        !!! Aborded
        """
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

    def get_nu_chunk(self):
        return self.num_chunk, self.num_val_chunk, self.num_test_chunk

    def reload_data(self, src_sys, src_sys2, src_ref, tgt, src_val_sys, src_val_sys2, src_val_ref, tgt_val, src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output):
        self.cur_index = 0
        self.cur_val_index = 0
        self.cur_test_index = 0
        self._load_data(src_sys, src_sys2, src_ref, tgt, src_val_sys, src_val_sys2, src_val_ref, tgt_val, src_test_sys, src_test_sys2, src_test_ref, tgt_test, rank, sf_output)

    def split_dataset(self, dat, chunk_size = 5000, cut = None, dim = 1, rm = True):
        """
        split the large dataset into small chunck and save it in the same dir with the suffix '_sub_num'
        input:
            dat: .npy data
                the dataset
            chunk_size:
                the size of each chunck
            cut: None or int
                only get the fisrt 'cut' value in the axis 'dim'
            remove: 
                weather remove the origin data or not
        """
        # read data and set attribute
        name_dir = dat
        suffix = '_sub_'
        dat = np.load(name_dir)
        dat = torch.from_numpy(dat)
        # do the cut
        if cut is not None:
            index = torch.arange(0, cut).type(torch.LongTensor)
            dat = torch.index_select(dat, dim, index)
        # split the data
        sp_dat = torch.split(dat, chunk_size)
        # save chunk data
        num_chunk = len(sp_dat)
        name_out = name_dir + 'suffix'
        for ind, chunk in enumerate(sp_dat):
            np.save(name_out+str(ind), chunk.numpy())
        # clean data
        if rm:
            os.system('rm '+ dir)

"""
    def get_batch_repeatly(self, sep = False):
        if self.cur_index == self.nu_batch:
            self.cur_index = 0
            self._shuffle()
        return self.get_batch(sep)
"""
