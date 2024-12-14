import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

hmp_data = pd.read_csv("./data/hmp/normalized_hmp_data.csv", index_col=0)
ibd_data = pd.read_csv("./data/ibdmdb/normalized_ibd_data.csv", index_col=0)
# take the mean of all columns and plot in bar graph
ibd_mean_values = ibd_data.mean()
ibd_mean_values.plot(kind='bar', label='IBD', color='red', alpha=0.7)
hmp_mean_values = hmp_data.mean()
hmp_mean_values.plot(kind='bar', label='HMP', color='blue', alpha=0.7)
plt.xlabel('Bacteria Genus')
plt.ylabel('Mean Value')
plt.title('Average Value of Bacteria Genus in Both Datasets')
plt.show()