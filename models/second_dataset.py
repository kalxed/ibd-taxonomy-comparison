import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re



def aggregate_genus(ibd_data):
    # Collapse the columns to genus level
    ibd_regex = "g__([^\.]*?)\."
    converted = []
    for column in ibd_data.columns[1:-1]:
        genus = re.findall(ibd_regex, column)
        if genus:
            if genus[0] in converted:
                ibd_data[genus[0]] = ibd_data[genus[0]] + ibd_data[column]
                ibd_data.drop(columns=[column], axis=1, inplace=True)
            else:
                ibd_data.rename(columns={column: genus[0]}, inplace=True)
                converted.append(genus[0])
        else:
            ibd_data.drop(columns=[column], axis=1, inplace=True)
    ibd_data.to_csv("./data/ibdmdb/ibd_pathabundance_genus_summed.csv")
    return ibd_data

# ibd_data = pd.read_csv("data/ibdmdb/qiime2_otu_with_taxonomy_original.csv", index_col=7)
# ibd_data.drop(["Unnamed: 0","Taxon","Domain","Phylum","Class","Order","Family","Species","Confidence"], axis=1, inplace=True)
# ibd_data.dropna(inplace=True, axis=0)
# ibd_data = ibd_data.transpose()
# # ibd_data.to_csv("data/ibdmdb/ibd_qiime_genus.csv")
# # Display the current column names

# ibd_summed = ibd_data.groupby(ibd_data.columns, axis=1).sum()
# print(ibd_summed)
# ibd_summed.to_csv("data/ibdmdb/ibd_qiime_genus_summed_original.csv")

# plot ibd_summed bar graph as mean of columns
final_ibd = pd.read_csv("data/ibdmdb/normalized_ibd_final.csv", index_col=0)
final_hmp = pd.read_csv("data/hmp/normalized_hmp_final.csv", index_col=0)

