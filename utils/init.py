import torch
import torch.nn as nn
import torch.nn.init as init

def weight_init(model):
    """
    initialize the parameters in the torch.nn module 
    parameters of the self defined model will be initialized in the __init__ method of each model
    """
    print 'initializing the model...'
    if isinstance(model, nn.Linear):
        if debug:
            print '=> initializing Linear'
        init.xavier_normal(model.weight)
        if model.bias:
            init.xavier_uniform(model.bias)
    elif isinstance(model, nn.BatchNorm1d):
        if debug:
            print '=> initializing BatchNorm1d'
        model.weight.data.fill_(1)
        model.bias.data.zero_()
    elif (isinstance(model, nn.LSTM) | isinstance(model, nn.GRU) | isinstance(model, nn.RNN)):
        if debug:
            print '=> initializing rnn'
        rnn_params = list(model.parameters())
        num_layers = len(rnn_params)/2
        for index, item in enumerate(rnn_params):
            if index < num_layers:
                init.orthogonal(rnn_params[index])
                init.orthogonal(rnn_params[index])
            else:
                init.xavier_uniform(rnn_params[index])
                init.xavier_uniform(rnn_params[index])
    elif isinstance(model, nn.Conv1d):
        if debug:
            print '=> initiliazing conv1d'
        init.xavier_normal(model.weight)
        init.xavier_uniform(model.bias)