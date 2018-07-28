from __future__ import division

import sys
import numpy
import argparse
parser = argparse.ArgumentParser(description='compare_hidden.py')


parser.add_argument('-s1', default="",
                    help="""Path to vector for s1""")
parser.add_argument('-s2', default="",
                    help="""Path to vector for s2""")
parser.add_argument('-ref', default="",
                    help="""Path to vector for ref""")

def main():
    opt = parser.parse_args()
    s1 = numpy.load(opt.s1);
    s2 = numpy.load(opt.s2);
    ref = numpy.load(opt.ref);

    s1 = numpy.sum(s1,1)
    s2 = numpy.sum(s2,1)
    ref = numpy.sum(ref,1)
    diff1 = numpy.sum(numpy.absolute(s1-ref),1)
    diff2 = numpy.sum(numpy.absolute(s2-ref),1)
    for i in range(diff1.shape[0]):
        if(diff1[i] > diff2[i]):
            print 1
        else:
            print -1
    

if __name__ == "__main__":
    main()

