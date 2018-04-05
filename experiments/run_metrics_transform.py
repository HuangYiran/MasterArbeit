# -*- coding: UTF-8 -*-
import sys
sys.path.append("./models/")
import torch
import argparse
import numpy
import match

from MetricsTransform import MetricsTransform

parser = argparse.ArgumentParser()
parser.add_argument('-met', default = 'beer', help = 'the metrics that use the get the metrics scores')

def main():
    opt = parser.parse_args()
    ##### load data, firstly, make sure the type of the input data, then read to data and translate it to the correct type
    # set file path
    file_src_train = '../data/MasterArbeit/metrics_transform/'+opt.met+'/src_train_scores'
    file_src_test = '../data/MasterArbeit/metrics_transform/'+opt.met+'/src_test_scores'
    file_tgt_train = '../data/MasterArbeit/metrics_transform/'+opt.met+'/tgt_train_scores'
    file_tgt_test = '../data/MasterArbeit/metrics_transform/'+opt.met+'/tgt_test_scores'

    # load data
    src_train = [float(li.rstrip('\n') for li in open(file_src_train))]
    src_test = [float(li.rstrip('\n') for li in open(file_src_test))]
    tgt_train = [float(li.rstrip('\n') for li in open(file_tgt_train))]
    tgt_test = [float(li.rstrip('\n') for li in open(file_tgt_test))]

    # change type
    sre_train = numpy.asarray(src_train)
    src_test = numpy.asarray(src_test)
    tgt_train = numpy.asarray(tgt_train)
    tgt_test = numpy.asarray(tgt_test)

    src_train = torch.from_numpy(src_train)
    src_test = torch.from_numpy(src_test)
    tgt_train = torch.from_numpy(tgt_train)
    tgt_test = torch.from_numpy(tgt_test)

    ##### build model 
    model = MetricsTransform()

    ##### set optimizer
    optim == torch.optim.Adam(model.parameters())

    ##### set loss
    loss_fn = torch.nn.MSELoss()

    ##### assert
    assert(len(src_test) == len(tgt_test))
    assert(len(src_train) == len(tgt_train))

    ##### train the model 
    ROUND = 200
    cur_index = 0
    batch_size = 50
    len_train_data = src_train.shape[0]
    len_test_data = src_test.shape[0]
    num_train_batch = len_train_data/batch_size
    num_test_batch = len_test_data/betch_size
    for i in range(ROUND):
        if cur_index == num_train_batch:
            cur_index = 0
        # get train data
        start = batch_size * cur_index
        end = batch_size + start
        if start > len_train_data:
            print("the data set is empty")
            src, tgt = None, None
        elif end > len_train_data:
            end = len_train_data
        cur_index += 1
        src, tgt = src_train[start:end,], tgt_train[start:end,]
        if i % 10 == 0:
        # evaluate if neccesary 
            loss = evaluate(model, loss_fn, src, tgt)
            print "the val loss is: " + loss
        else:
            # train
            train_batch(model, loss_fn, optim, src, tgt)

    ##### test the model
    loss = 0
    for i in len(num_test_data):
        loss += evaluate(model, loss_fn, src, tgt)
    loss /= num_test_data*1.0
    
    ##### return the result
    return loss

def evaluate(model, looss_fn, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    model.eval()
    out = model(src)
    loss = loss_fn(out, tgt).data
    loss = loss.numpy()
    mean = numpy.mean(loss)
    return mean

def train_batch(model, loss_fn, optim, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    model.train()
    out = model(src)
    loss = loss_fn(out, tgt)
    loss.backward()
    optim.step()
    return True
if __name__ == "__main__":
    main()
