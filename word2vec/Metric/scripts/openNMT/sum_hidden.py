from __future__ import division

import sys
import numpy
import argparse
parser = argparse.ArgumentParser(description='compare_hidden.py')


parser.add_argument('-input', default="",
                    help="""Path to vector for s1""")
parser.add_argument('-output', default="",
                    help="""Path to vector for s2""")

def main():
    opt = parser.parse_args()
    s1 = numpy.load(opt.input);
    s1 = numpy.sum(s1,1)

    with open(opt.output, "w") as f:
        numpy.save(f, s1)

    

if __name__ == "__main__":
    main()

