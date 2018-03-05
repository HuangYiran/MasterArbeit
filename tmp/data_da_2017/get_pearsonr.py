from scipy import stats

l1 = [float(li.rstrip('\n')) for li in open("./tmp/data_da_2017/data_scores")]
l2 = [float(li.rstrip('\n')) for li in open("./tmp/data_da_2017/chrF_scores_clean")]
l2.pop(0)
l2.pop(-1)
l2.pop(-1)
l2.pop(-1)

pearsonr = stats.pearsonr(l1, l2)
print pearsonr
