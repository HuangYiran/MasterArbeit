#-*- coding:UTF-8 -*-
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csv', default = "./collect_data/wmt16_rank_deen.csv", help = "the csv file to extract the scores")
parser.add_argument('-sys', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt16-metrics-task/sys", help = "the dir to save the system output file")
parser.add_argument('-ref', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt16-metrics-task/ref/newstest2016-deen-ref.en", help = "the reference file")
parser.add_argument('-src', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt16-metrics-task/source/newstest2016-deen-src.de", help = "the source file")

def main():
    opt = parser.parse_args()
    # extract the scores data from the csv file
    r = "Rscript ./collect_data/extract_rr_data_from_csv.r " + opt.csv
    os.system(r)
    # read the nme of the system
    names = [li.rstrip('\n')[1:-1] for li in open("./tmp/tmp_system_names")]
    print names
    # get the filename of the system output
    walk = os.walk(opt.sys)
    filenames = ""
    for root, dir, fname in walk:
        filenames = fname
    # read all the sentences to the dict attribute
    dict_sys = {}
    for filename in filenames:
        dict_sys[filename] = add_one(opt.sys + "/" + filename)
    # read the source, reference and score data
    con_ref = add_one(opt.ref)
    con_src = add_one(opt.src)
    # attributes to save the data
    data_src = []
    data_ref = []
    data_s1 = []
    data_s2 = []
    data_result = []
    # method to add the data
    def add_data(src, ref, s1, s2, result):
        data_src.append(src)
        data_ref.append(ref)
        data_s1.append(s1)
        data_s2.append(s2)
        data_result.append(result)
    # read data
    print "start reading data from ./tmp/extract_data"
    with open('./tmp/extract_rank_deen') as fi:
        for line in fi:
            tmp = line.split(',')
            # cols of tmp: 'sid', 's1', 's2', 'result'
            # pay attention to that, the sid begin with 1 but the index is zero 
            sid = int(tmp[0].strip())
            name_s1 = tmp[1][1: -1]
            name_s2 = tmp[2][1: -1]
            sen_src = con_src[sid - 1]
            sen_ref = con_ref[sid - 1]
            sen_s1 = dict_sys[name_s1][sid - 1]
            sen_s2 = dict_sys[name_s2][sid - 1]
            sen_result = tmp[3]
            add_data(sen_src, sen_ref, sen_s1, sen_s2, sen_result)
    print "<<< finish reading data"
    # write data
    print "start writing the data to the file"
    write_data("./collect_data/data_s1", data_s1)
    write_data("./collect_data/data_s2", data_s2)
    write_data("./collect_data/data_ref",data_ref)
    write_data("./collect_data/data_source", data_src)
    write_data("./collect_data/data_scores", data_result)
    print "<<<finishing writing the data"

    # remove tmp data
    print "cleaning the tmp data"
    #os.system("rm ./tmp/*")
    print "<<<finish cleaning the tmp data"

def add_one(file):
    tmp = [li.rstrip('\n') for li in open(file)]
    return tmp

def get_filename(name, filelist):
    index = 0
    while(index < len(filelist)):
        if name in filelist[index]:
            return filelist[index]
        index = index + 1
    if index == len(filelist):
        print("file not exist!!")
        return "file_does_not_exist"

def write_data(filename, li):
        with open(filename, 'w') as fi:
            for line in li:
                if isinstance(line, float) or isinstance(line, int):
                    line = str(line)+'\n'
                fi.write(line)


if __name__ == "__main__":
    main()
