"""
attention:
    the src file should order by sid, 
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-src', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_a_source_en.de/extracted_data/data_source",
        help = "the source file")
parser.add_argument('-scores', default= "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_a_source_en.de/extracted_data/data_scores",
        help = "the da scores file")
parser.add_argument('-tgt', default = "/Users/ihuangyiran/Documents/Workplace_Python/data/MasterArbeit/plan_a_source_en.de/extracted_data/darr",
        help = "the file to write the result")
parser.add_argument('-threshold', default = 10, help = "the difference to distinct two da scores")

def horToVer(li, threshold = 25, index = 0):
    length = len(li)
    tmp = []
    for i in range(length):
        j = i + 1
        while j < length:
            if float(li[i]) - float(li[j]) >= threshold:
                rr = 1
            elif float(li[i]) - float(li[j]) <= -threshold:
                rr = -1
            else:
                rr = 0
#            if rr == 0:
#                j += 1
#                continue
            tmp.append(str(index) + "," + li[i] + "," + li[j] + "," + str(rr))
            j += 1
    return tmp

opt = parser.parse_args()

# load the data
data_src = [li.rstrip('\n') for li in open(opt.src)]
data_scores = [li.rstrip('\n') for li in open(opt.scores)]
# set attributes
darr = [] # list of string, sid, s1, s2, rr
sid = 1
num_repeat = 1 # how many time is the sentence repeated
current_sent = data_src.pop(0)
scores_batch = [] # store the scores of the same sentence
scores_batch.append(data_scores.pop(0))
# convert the data
for (src, score) in zip(data_src, data_scores):
    if src == current_sent:
        num_repeat += 1
    else:
        if num_repeat > 1:
            darr.extend(horToVer(scores_batch, threshold = float(opt.threshold), index = sid))
        num_repeat = 1
        current_sent = src
        scores_batch = []
        sid += 1
    scores_batch.append(score)
if len(scores_batch) > 1:
    darr.extend(horToVer(scores_batch, threshold = float(opt.threshold), index = sid))
# write the data
o = open(opt.tgt, 'w')
for li in darr:
    o.write(li)
    o.write('\n')
o.close()

