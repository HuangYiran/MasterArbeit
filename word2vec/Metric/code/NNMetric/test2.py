"""
agged data are given 
here we use different distance methods to compute the distance between two sentence
"""
import argparse
import numpy as np

parser = argparse.ArgumentParser(description= 'test2.py')
parser.add_argument('-ref', required = True, help = 'Reference file')
parser.add_argument('-hyp', required = True, help = 'Hypothesis')
parser.add_argument('-type', required = True, help = 'distance methods')

def L1(v1, v2):
    #return np.absolute(v1-v2).sum()
    return np.linalg.norm(v1-v2, ord = 1)

def L2(v1, v2):
    return np.linalg.norm(v1-v2)

def cos(v1, v2):
    mul = np.dot(v1,v2)
    n1  = np.linalg.norm(v1)
    n2 = np. linalg.norm(v2)
    return mul*1./(n1*n2)

def main():
    opt = parser.parse_args()

    rf = np.load(opt.ref)
    hf = np.load(opt.hyp)

    for i,j in zip(rf,hf):
        if opt.type == 'L1':
            print ("Distance:", L1(i,j))
        elif opt.type == 'L2':
            print ("Distance:", L2(i,j))
        elif opt.type == 'cos':
            print ("Distance:", cos(i,j))

if __name__ == "__main__":
    main()
