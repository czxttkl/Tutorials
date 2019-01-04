import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="darkgrid")

df_dict = {
        'epochs': [0, 0, 0, 1, 1, 2, 2, 3, 3],
        'event': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'signal': [1, 2, 3, 4, 5, 6, 7, 8, 11],
    }
df = pd.DataFrame(df_dict)

# Plot the responses for different events and regions
my_plot = sns.lineplot(x="epochs", y="signal",
                      # hue="event", style="event",
                       data=df).set_title("LaLaLa")
my_plot.get_figure().savefig("seaborn_plot.png")
plt.show()


