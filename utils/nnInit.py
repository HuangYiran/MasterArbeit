import torch
import torch.nn as nn
import torch.nn.init as init

def weight_init(model):
    """
    initialize the parameters in the torch.nn module 
    parameters of the self defined model will be initialized in the __init__ method of each model
    """
    if isinstance(model, nn.Linear):
        print '=> initializing Linear model'
        print model
        init.xavier_normal(model.weight)
        if model.bias is not None:
            num_biases = len(model.bias)
            model.bias.data.fill_(1.0/num_biases)
    elif isinstance(model, nn.BatchNorm1d):
        print '=> initializing BatchNorm1d'
        print model
        model.weight.data.fill_(1)
        model.bias.data.zero_()
    elif (isinstance(model, nn.LSTM) | isinstance(model, nn.GRU) | isinstance(model, nn.RNN)):
        print '=> initializing rnn'
        print model
        rnn_params = list(model.parameters())
        num_layers = len(rnn_params)/2
        print("num layers: ", num_layers)
        for index in range(num_layers):
            if index % 2 == 0:
                init.orthogonal(rnn_params[2 * index])
                init.orthogonal(rnn_params[2 * index + 1])
            else:
                num_biases = len(rnn_params[2 * index])
                rnn_params[2 * index].data.fill_(1.0/num_biases)
                num_biases = len(rnn_params[2 * index + 1])
                rnn_params[2 * index + 1].data.fill_(1.0/num_biases)
    elif isinstance(model, nn.Conv1d):
        print '=> initiliazing conv1d'
        print model
        init.xavier_normal(model.weight)
        num_biases = len(model.bias)
        model.bias.data.fill_(1.0/num_biases)
    else:
        print '=> spring the self defined model'
        print model
