tmp = [li.split('\t')[-1] for li in open('./tmp/data_da_2017/chrF_scores')]
with open('./tmp/data_da_2017/chrF_scores_clean', 'w') as fi:
    for li in tmp:
        fi.write(li)
