#-*- coding: utf-8 -*-

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

def plot_mid_result(dic, name = 'plot', o_dir = '/tmp/test.png'):
    """
    plot the mid result during the training of the model, and save the plot 
    input:
        dic: type of dict, store the value that want to be plotted. value must all same length
        name: set the name of the plot
        o_dir: path the save the target plot
    """
    # create pd.DataFrame from the dict, and conert it to stack mode
    df = pd.DataFrame.from_dict(dic).reset_index()
    df_conv = pd.melt(df, id_vars = ['index'], value_vars = list(dic.keys()), var_name = 'type', value_name = 'value')
    # plot and save
    g = sns.lmplot(x = 'index', y = 'value', hue = 'type', data = df_conv, scatter_kws={"s": 10})
    g.fig.suptitle(name)
    g.savefig(o_dir)

