import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_seaborn(df, xaxis, yaxis, title, file, show):
    sns.set(style="darkgrid")

    my_plot = sns.lineplot(x=xaxis, y=yaxis, data=df).set_title(title)
    my_plot.get_figure().savefig(file)
    if show:
        plt.show()
