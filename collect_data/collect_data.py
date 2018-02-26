#-*- coding:UTF-8 -*-
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-csv', default = "./collect_data/ad-seg-scores-de-en.csv", help = "the csv file to extract the scores")
parser.add_argument('-sys', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt17-metrics-task/wmt17-submitted-data/txt/system-outputs/newstest2017/de-en", help = "the dir to save the system output file")
parser.add_argument('-ref', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt17-metrics-task/wmt17-submitted-data/txt/references/newstest2017-deen-ref.en", help = "the reference file")
parser.add_argument('-src', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/wmt17-metrics-task/wmt17-submitted-data/txt/sources/newstest2017-deen-src.de", help = "the source file")

def main():
    opt = parser.parse_args()
    # extract the scores data from the csv file
    r = "Rscript ./collect_data/extract_data_from_csv.r "+ opt.csv
    os.system(r)

    # read the name of the system
    names = [li.rstrip('\n')[1:-1] for li in open("./tmp/tmp_system_names")]

    print names
    # get the filename of the system output
    walk = os.walk(opt.sys)
    filenames = ""
    for root, dir, fname in walk:
        filenames = fname

    # attributes to save the data
    data_sys_out = []
    data_ref = []
    data_source = []
    data_scores = []
    
    # method to add the data
    def add_data(li_r, li_s, li_o, score):
        data_ref.append(li_r)
        data_source.append(li_s)
        data_sys_out.append(li_o)
        data_scores.append(score)

    # read data
    for sysname in names:
        # 读取ref和src文件
        file_ref = add_one(opt.ref)
        file_src = add_one(opt.src)
        # 读取对应的sys_out文件，要求是名字包含有sysname的对应的文件
        filename = get_filename(sysname, filenames)
        if filename == 'file_does_not_exist':
            continue
        file_sys_out = add_one(opt.sys + "/" + filename)
        cfile = zip(file_ref, file_src, file_sys_out)

        # 读取analyse文件，并把行数和分数分别存在列表sids和tmp_scores中
        file_ana = open("tmp/" + sysname)
        sids = []
        tmp_scores = []
        for line in file_ana:
            tmp = line.split(",")
            sids.append(tmp[0])
            tmp_scores.append(tmp[1])

        assert(len(sids) == len(tmp_scores))
        print("start extract data from file " + "./test_data/" + sysname)
        print(sysname + " file long:", len(sids))
        index_sent = 1 # 指明当前看的是文档中的第几个句子，从1开始计数
        index_ana = 0 # 指明当前看的是ana文件中的第几项，从0开始计数
        end = False
        for item in cfile:
            # 因为文件是按照sid进行排序的，所以可以这么处理
            li_r = item[0]
            li_s = item[1]
            li_o = item[2]
            # 如果sid该项所看的句子，刚好是现在所在的句子，就把这个句子记起来，否则跳过
            # 可能存在一个句子有多个人打分的情况，所以这里用while，而不是用if
            # 因为不存在无用的项，所以仅在保存了该项的内容后，才会增加index_ana
            sid = sids[index_ana]
            while str(index_sent) == sid:
                add_data(li_r, li_s, li_o, tmp_scores[index_ana])
                index_ana = index_ana + 1
                if index_ana < len(sids):
                    sid = sids[index_ana]
                else:
                    # 该系统的分析文件已经读完了，所以就结束程序吧
                    end = True
                    break
            # 假如index_sent和sid不一致，说明ana当前项看的并不是这个句子，所以跳到下一个句子
            index_sent = index_sent + 1
            if end:
                break

        print("..end extract data from file " + "./test_data/" + sysname)

        file_ref.close()
        file_src.close()
        file_sys_out.close()
        file_ana.close()

    # write data
    write_data("./collect_data/data_sys_out", data_sys_out)
    write_data("./collect_data/data_ref",data_ref)
    write_data("./collect_data/data_source", data_source)
    write_data("./collect_data/data_scores", data_scores)

    # remove tmp data
    os.system("rm ./tmp/*")

def add_one(file):
    tmp = open(file)
    for line in tmp:
        yield line
    yield None

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
                fi.write(line)

if __name__ == "__main__":
    main()
