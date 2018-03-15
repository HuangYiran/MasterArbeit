#-*- coding:UTF-8 -*-
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csv', default = "./collect_data/wmt14-deen.csv", help = "the csv file to extract the scores")
parser.add_argument('-sys', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/de-en_2014/sys", help = "the dir to save the system output file")
parser.add_argument('-ref', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/de-en_2014/ref/newstest2014-deen-ref.en", help = "the reference file")
parser.add_argument('-src', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_c_source_rank_de.en/de-en_2014/src/newstest2014-deen-src.de", help = "the source file")
parser.add_argument('-srclang', default = 'German')
parser.add_argument('-trglang', default = 'English')

def main():
    opt = parser.parse_args()
    with open(opt.csv) as fi:
        length = len(fi.readline().split(','))
        if length > 20:
            r = "Rscript ./collect_data/convert_rr5_to_rr2.r " + opt.csv + " " + opt.srclang + " " + opt.trglang
            print r
            os.system(r)
            r = "Rscript ./collect_data/extract_rr_data_from_csv.r ./tmp/tmp.wmt14.stacked.csv " + opt.srclang + " " + opt.trglang
        else:
            # extract the scores data from the csv file
            r = "Rscript ./collect_data/extract_rr_data_from_csv.r " + opt.csv + " " + opt.srclang + " " + opt.trglang 
    print r
    os.system(r)
    # get the mode value for the same sid+sys pair
    r = "Rscript ./collect_data/get_mode_for_rr.r ./tmp/extracted_deen_2016"
    print r
    os.system(r)
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
    with open('./tmp/extracted_deen_2016') as fi:
        for line in fi:
            tmp = line.split(',')
            # cols of tmp: 'sid', 's1', 's2', 'result'
            # pay attention to that, the sid begin with 1 but the index is zero 
            sid = int(tmp[0].strip())
            #name_s1 = tmp[1][1: -1]
            #name_s2 = tmp[2][1: -1]
            name_s1 = tmp[1][1:-1]
            name_s2 = tmp[2][1:-1]
            sen_src = con_src[sid - 1]
            sen_ref = con_ref[sid - 1]
            if name_s1.split('.')[-1] == "txt":
                name_s1 = name_s1[:-4]
            if name_s2.split('.')[-1] == "txt":
                name_s2 = name_s2[:-4]
            name_s1 = get_filename(name_s1, filenames)
            name_s2 = get_filename(name_s2, filenames)
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
    tmp = [li for li in open(file)]
    return tmp

def get_filename(name, filelist):
    index = 0
    while(index < len(filelist)):
        if name in filelist[index]:
            return filelist[index]
        index = index + 1
    if index == len(filelist):
        print("file not exist!!")
        return "file_does_not_exist: "+ name

def write_data(filename, li):
        with open(filename, 'w') as fi:
            for line in li:
                if isinstance(line, float) or isinstance(line, int):
                    line = str(line)+'\n'
                fi.write(line)


if __name__ == "__main__":
    main()
