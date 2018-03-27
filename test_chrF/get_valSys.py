import os
import argparse

from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('-lp', default = 'deen', help = "the directioin of the translation")
parser.add_argument('-da', action = 'store_true', default = False, help = "da can only be used in deen direction")


opt = parser.parse_args()
lp = opt.lp
srclang = lp[:2]
trglang = lp[2:]

if opt.da:
    assert(opt.lp == 'deen')

path = './system_level/'+ lp + '_2016/'
filenames = os.listdir(path+'sys/')

sys_names = []
if os.path.exists('./system_level/'+ lp +'2016.txt'):
    rm = 'rm ./system_level/'+lp+'2016.txt'
    os.system(rm)
for filename in filenames:
    tmp = filename.split('.')
    data = tmp[0]
    d1 = filename.find('.')+1
    d2 = filename.rfind('.')
    sys_name = filename[d1:d2]
    ref = path+'ref/newstest2016-'+lp+'-ref.'+trglang
    sys_names.append(sys_name)
    command = 'chrF++.py -R ' + ref + ' -H ' + path+'sys/'+filename + " -nw 0 -b 2 "
    command_plus = "| awk 'BEGIN{counter = 0}{if(counter == 1){print $2};counter++;}'>>system_level/" + lp + "2016.txt"
    os.system(command+command_plus)

if not opt.da:
    # read human assessment
    human_scores = {}
    with open('./system_level/human_'+lp+'_2016.txt') as fi:
        for li in fi:
            tmp = li.split('\t')
            sys_name = tmp[0]
            human_score = float(tmp[1].rstrip('\n'))
            if sys_name not in human_scores.keys():
                human_scores[sys_name] = human_score

    # reorder the human assessment according to the chrF output
    human_scores_reorder = []
    for sys in sys_names:
        for key in human_scores.keys():
            if sys in key:
                human_scores_reorder.append(human_scores[key])

    # read chrF output
    chrF_scores = [float(li) for li in open('./system_level/'+lp+'2016.txt')]

    print len(sys_names)
    print sys_names
    print chrF_scores
    print human_scores_reorder
    # get the correaltion
    pearsonr = stats.pearsonr(human_scores_reorder, chrF_scores)
    print pearsonr
else:
    # da
    # read human assessment
    human_scores = {}
    with open('./system_level/human_da_deen_2016.txt') as fi:
        for li in fi:
            tmp = li.split(' ')
            sys_name = tmp[0]
            human_score = float(tmp[1].rstrip('\n'))
            if sys_name not in human_scores.keys():
                human_scores[sys_name] = human_score

    # reorder the human assessment according to the chrF output
    sys_names = [li.split('.')[0] for li in sys_names]
    human_scores_reorder = []
    for sys in sys_names:
        for key in human_scores.keys():
            if sys in key:
                human_scores_reorder.append(human_scores[key])

    # read chrF output
    chrF_scores = [float(li) for li in open('./system_level/'+lp+'2016.txt')]

    print len(sys_names)
    print sys_names
    print chrF_scores
    print human_scores_reorder
    # get the correaltion
    pearsonr = stats.pearsonr(human_scores_reorder, chrF_scores)
    print pearsonr

