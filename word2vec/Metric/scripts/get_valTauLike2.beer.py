import argparse
import os
import sys
sys.path.append('.test_beer/beer_2.0')
sys.path.append('./utils')
from valuation import valTauLike
from normalize import normalize_minmax
from transform import daToRr

parser = argparse.ArgumentParser()
parser.add_argument('-s1', default = "./test_beer/data_s1", help = "the first system output file")
parser.add_argument('-s2', default = "./test_beer/data_s2", help = "the second system output file")
parser.add_argument('-ref', default = "./test_beer/data_ref", help = "the reference file")
parser.add_argument('-src', default = "./test_beer/data_src", help = "the source file")
parser.add_argument('-scores', default = "./test_beer/data_scores", help = "The file that store the result")
parser.add_argument('-threshold', default = 0.1, help = "The threshold to distinguish two score")

def main():
    # load the parameter from the command line
    opt = parser.parse_args()
    # run the beer to get the beer scores for s1 and s2
    beer1 = "./code/beer_2.0/beer -s " + opt.s1 + " -r " + opt.ref + " --printSentScores > ./tmp/tmp1"
    beer2 = "./code/beer_2.0/beer -s " + opt.s2 + " -r " + opt.ref + " --printSentScores > ./tmp/tmp2"
    print beer1
    print beer2
    os.system(beer1)
    os.system(beer2)
    # extract the score form the file and remove the last line of the tmp file, it contains the total valuation
    tmp_sc1 = [float(li.strip('\n').split(' ')[-1]) for li in open('./tmp/tmp1')]
    tmp_sc1.pop(-1)
    tmp_sc2 = [float(li.strip('\n').split(' ')[-1]) for li in open('./tmp/tmp2')]
    tmp_sc2.pop(-1)
    # transform the da data to rr data, darr1 is the human scores
    darr = daToRr(tmp_sc1, tmp_sc2, float(opt.threshold))
    writefile(darr, './tmp/darr')
    # read the human socres
    rr = [int(li.rstrip('\n')) for li in open(opt.scores)]
    taur = valTauLike(rr, darr)
    writefile(taur, './tmp/taur', 'a')
    print taur

def writefile(fi, name, mode = 'w'):
    with open(name, mode) as out:
        if isinstance(fi, float):
            out.write(str(fi)+'\n')
        else:
            for li in fi:
                if isinstance(li, float) or isinstance(li, int):
                    li = str(li) + '\n'
                out.write(li)

if __name__ == "__main__":
    main()
