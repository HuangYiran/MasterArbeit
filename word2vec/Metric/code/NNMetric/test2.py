import argparse
import numpy as np

parser = argparse.ArgumentParser(description= 'test2.py')
parser.add_argument('-ref', required = True, help = 'Reference file')
parser.add_argument('-hyp', required = True, help = 'Hypothesis')

def distance(v1, v2):
    return np.absolute(v1-v2).sum()

def main():
    opt = parser.parse_args()

    rf = np.load(opt.ref)
    hf = np.load(opt.hyp)

    for i,j in zip(rf,hf):
        print ("Distance:", distance(i,j))

if __name__ == "__main__":
    main()
