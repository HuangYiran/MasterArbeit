# -*- coding: UTF-8 -*-
import sys
sys.path.append("./models/")
import torch
import argparse
import numpy
import math

from scipy import stats
from MetricsTransform import MetricsTransform

parser = argparse.ArgumentParser()
parser.add_argument('-met', default = 'beer', help = 'the metrics that use the get the metrics scores')
parser.add_argument('--transform', action = 'store_true', help = 'enter predict model or not')
parser.add_argument('-checkpoint', default = 'checkpoints/transform.cp', help = 'the file that save the checkpoint')
parser.add_argument('-src', default = '../data/MasterArbeit/plan_c_source_da_de.en/extracted_data_sub_metrics/metrics/beer/data_scores', help = 'if predict, we need this file to store the source file')
parser.add_argument('-out', default = '../data/MasterArbeit/plan_c_da_de.en/sub_accuracy/metrics/data_scores', help = 'if predict, save the output here')

def main():
    opt = parser.parse_args()
    ##### load data, firstly, make sure the type of the input data, then read to data and translate it to the correct type
    # set file path
    file_src_train = '../data/MasterArbeit/metrics_transform/'+opt.met+'/src_train_scores'
    file_src_test = '../data/MasterArbeit/metrics_transform/'+opt.met+'/src_test_scores'
    file_tgt_train = '../data/MasterArbeit/metrics_transform/'+opt.met+'/tgt_train_scores'
    file_tgt_test = '../data/MasterArbeit/metrics_transform/'+opt.met+'/tgt_test_scores'

    # load data
    src_train = [float(li.rstrip('\n')) for li in open(file_src_train)]
    src_test = [float(li.rstrip('\n')) for li in open(file_src_test)]
    tgt_train = [float(li.rstrip('\n')) for li in open(file_tgt_train)]
    tgt_test = [float(li.rstrip('\n')) for li in open(file_tgt_test)]

    # change type
    src_train = numpy.asarray(src_train)
    src_test = numpy.asarray(src_test)
    tgt_train = numpy.asarray(tgt_train)
    tgt_test = numpy.asarray(tgt_test)

    src_train = torch.from_numpy(src_train).unsqueeze(dim =1).float()
    src_test = torch.from_numpy(src_test).unsqueeze(dim = 1).float()
    tgt_train = torch.from_numpy(tgt_train).unsqueeze(dim = 1).float()
    tgt_test = torch.from_numpy(tgt_test).unsqueeze(dim = 1).float()

    ##### build model 
    model = MetricsTransform()

    ##### set optimizer
    optim = torch.optim.Adam(model.parameters())

    ##### if transform, transfrom the data and save it
    if opt.transform:
        print ('load parameters from the "{}"'.format(opt.checkpoint))
        checkpoint = torch.load(opt.checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        # load data
        print 'load src data'
        src = [float(li.rstrip('\n')) for li in open(opt.src)]
        src = numpy.asarray(src)
        src = torch.from_numpy(src).unsqueeze(dim = 1).float()
        # transform
        print 'transform the data'
        out = predict(model, src)
        # save the output
        print 'save the data'
        save_file(opt, out)
        return True

    ##### set loss
    loss_fn = torch.nn.MSELoss()

    ##### assert
    assert(len(src_test) == len(tgt_test))
    assert(len(src_train) == len(tgt_train))

    ##### train the model 
    ROUND = 200
    cur_index = 0
    batch_size = 100
    len_train_data = src_train.shape[0]
    len_test_data = src_test.shape[0]
    num_train_batch = len_train_data/batch_size
    num_test_batch = len_test_data/batch_size
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
            print "the val loss in step " + str(i) + ": " + str(loss)
        else:
            # train
            train_batch(model, loss_fn, optim, src, tgt)

    ##### save a checkpoint
    torch.save({
        'model': 'MetricsTransform',
        'state_dict': model.state_dict(),
        'optimizer': optim.state_dict()
        }, opt.checkpoint)
    ##### test the model
    loss = evaluate(model, loss_fn, src_test, tgt_test)

    pred = predict(model, src_test)
    
    # calculate the corr
    pred = pred.data.squeeze().numpy().tolist()
    src = src_test.squeeze().numpy().tolist()
    tgt = tgt_test.squeeze().numpy().tolist()
    print "correlation between src and tgt is: " + str(stats.pearsonr(src, tgt))
    print "correlation between pred and tgt is: " + str(stats.pearsonr(pred, tgt))
    print "test loss: " + str(loss)
    ##### return the result
    return loss

def evaluate(model, loss_fn, src, tgt):
    src = torch.autograd.Variable(src, requires_grad = False)
    tgt = torch.autograd.Variable(tgt, requires_grad = False)
    model.eval()
    out = model(src)
    loss = loss_fn(out, tgt).data
    loss = loss.numpy()
    mean = numpy.mean(loss)
    return mean

def predict(model, src):
    src = torch.autograd.Variable(src, requires_grad = False)
    out = model(src)
    return out

def save_file(opt, out):
    """
    the out data is coming from the predcit function
    the type of out is: torch.autograd.Variable
    """
    out = out.squeeze().data.numpy().tolist()
    with open(opt.out, 'w') as fi:
        for item in out:
            fi.write(str(item))
            fi.write('\n')

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
