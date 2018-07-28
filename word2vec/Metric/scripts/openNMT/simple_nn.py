from __future__ import division

import sys
import numpy
import argparse
parser = argparse.ArgumentParser(description='compare_hidden.py')
import torch
from random import shuffle

parser.add_argument('-s1', default="",
                    help="""Path to vector for s1""")
parser.add_argument('-s2', default="",
                    help="""Path to vector for s2""")
parser.add_argument('-ref', default="",
                    help="""Path to vector for ref""")

parser.add_argument('-valid_s1', default="",
                    help="""Path to vector for s1""")
parser.add_argument('-valid_s2', default="",
                    help="""Path to vector for s2""")
parser.add_argument('-valid_ref', default="",
                    help="""Path to vector for ref""")


parser.add_argument('-label', default="",
                    help="""Path to label""")

parser.add_argument('-valid_label', default="",
                    help="""Path to label""")


def load(s1,s2,ref,label_file):
    s1 = numpy.load(s1);
    s2 = numpy.load(s2);
    ref = numpy.load(ref);

    inp = numpy.concatenate((s1,s2,ref), axis=1)
    


    print inp.shape
    
    label = []
    with open(label_file, "r") as f:
        l = f.readline()
        while(l):
            label.append(int(l.strip()) + 1);
            l = f.readline();
    
    label = numpy.asarray(label)
    print "Label:",label.shape

    joint = [(inp[i],label[i]) for i in range(len(label))]
    shuffle(joint)
    inp = numpy.asarray([joint[i][0] for i in range(len(joint))])
    label = numpy.asarray([joint[i][1] for i in range(len(joint))])

    batchSize = 400

    inp = numpy.array_split(inp,batchSize,0)
    label = numpy.array_split(label,batchSize,0)


    return inp,label


def main():
    opt = parser.parse_args()


    inp,label = load(opt.s1,opt.s2,opt.ref,opt.label)
    dev_inp,dev_label = load(opt.valid_s1,opt.valid_s2,opt.valid_ref,opt.valid_label)

    #prepare data
                        

    print len(inp),inp[0].shape,label[0].shape


    model = torch.nn.Sequential(
          torch.nn.Linear(1500, 500),
          torch.nn.Tanh(),
          torch.nn.Linear(500, 3),    
#          torch.nn.Linear(1500,3),
          torch.nn.LogSoftmax()
        )
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss()

    calcBaseline(inp,label)
    calcError(inp,label,model)
    calcError(dev_inp,dev_label,model)


    i = 0

    epoch = 100

    for e in range(epoch):

        for i in range(len(inp)):
        
            x = torch.autograd.Variable(torch.from_numpy(inp[i]))
            y = torch.autograd.Variable(torch.from_numpy(label[i]))

            y_pred = model(x)

            # Compute and print loss
            loss = criterion(y_pred, y)
            #print(i, loss)
            
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    

        calcError(inp,label,model)
        calcError(dev_inp,dev_label,model)
    


def calcError(inp,label,model):


    nom = 0;
    sum = 0;

    for i in range(len(inp)):
        
        x = torch.autograd.Variable(torch.from_numpy(inp[i]))
        y = torch.autograd.Variable(torch.from_numpy(label[i]))

        y_pred = model(x)
        m, index = y_pred.max(1)
        

        for j in range(len(label[i])):
            if(label[i][j] != 1):
                nom += 1
            sum += (label[i][j]-1) * (index.data[j]-1)
    print "Result:"
    print sum,nom, 1.0*sum/nom



def calcBaseline(inp,label):


    nom = 0;
    sum = 0;

    for i in range(len(inp)):
        
        x = numpy.array_split(inp[i],3,1)
        diff = numpy.sign(numpy.sum(numpy.absolute(x[0]-x[2]),1)-numpy.sum(numpy.absolute(x[1]-x[2]),1))


        for j in range(len(label[i])):
            if(label[i][j] != 1):
                nom += 1
            sum += (label[i][j]-1) * diff[j]
    print "Baseline Result:"
    print sum,nom, 1.0*sum/nom


if __name__ == "__main__":
    main()

