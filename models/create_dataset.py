import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import re

def normalize_rows(dataframe):
    row_min = np.min(dataframe, axis=1, keepdims=True)
    row_max = np.max(dataframe, axis=1, keepdims=True)
    normalized = (dataframe - row_min) / np.where(row_max - row_min == 0, 1, row_max - row_min)
    return normalized

def load_ibd(filepath):
    # Loading, transposing, and adding labels to IBDMDB dataset
    ibd_data = pd.read_csv("./data/ibdmdb/taxonomic_profiles.tsv", sep="\t")
    taxonomies = ibd_data[["#OTU ID","taxonomy"]]
    ibd_data.drop(columns=["taxonomy"], inplace=True)
    ibd_data.set_index("#OTU ID", inplace=True)
    ibd_data = ibd_data.transpose()
    ibd_data["Label"] = "IBD"
    ibd_data.to_csv("./data/ibdmdb/ibd_pathabundance.csv")
    return ibd_data, taxonomies

# Loading, transposing, and adding labels to HMP dataset
def load_bulk_hmp(filepath):
    hmp_data = pd.read_csv(filepath, sep="\t", index_col=0)
    hmp_data = hmp_data.transpose()
    reference = pd.read_csv("hmp_manifests/HMIWGS_healthy.csv")
    hmp_data = hmp_data[hmp_data["SRS"].isin(reference["SRS ID"])]
    hmp_data = hmp_data[hmp_data["STSite"] == "Stool"]
    hmp_data["Label"] = "Healthy"
    hmp_data.to_csv("./data/hmp/healthy_pathabundance.csv")
    return hmp_data

def load_hmp(filepath):
    hmp_data = pd.read_csv(filepath)
    pattern = r"biosynthesis\|(.+)$"
    # Find matching columns
    matching_columns = [col for col in hmp_data.columns if pd.Series(col).str.contains(pattern, regex=True).any()]
    # Parse out the taxonomy name
    return hmp_data, matching_columns

def collapse_genus(hmp_data):
    # Collapse the columns to genus level
    hmp_regex = "g__([^\.]*?)\."
    for column in hmp_data.columns[9:-1]:
        genus = re.findall(hmp_regex, column)
        if genus:
            hmp_data.rename(columns={column: genus[0]}, inplace=True)
            
        else:
            hmp_data.drop(columns=[column], axis=1, inplace=True)
    print(hmp_data.columns)
    hmp_data = hmp_data.transpose()
    hmp_data.to_csv("./data/hmp/healthy_pathabundance_genus.csv")
    
    # summed_hmp = hmp_data[hmp_data.columns[9:-1]].sum(axis=1)
    # print(summed_hmp)
    # summed_hmp.to_csv("./data/hmp/healthy_pathabundance_genus_summed.csv")
    # return summed_hmp

def ibd_tax_parsing(hmp_data, hmp_taxonomies, ibd_taxonomies):
    # Find common taxonomies between HMP and IBDMDB datasets using fuzzy matching
    matched_tax = []
    ibd_taxonomies = list(ibd_taxonomies["taxonomy"])
    for ibd_tax in ibd_taxonomies:
        ibd_regex = "__([^;]*?)(?:;|$)"
        found_match = False
        ibd_tax_granularities = re.findall(ibd_regex,ibd_tax)
        print(ibd_tax_granularities, found_match)
        for ibd_gran_tax in reversed(ibd_tax_granularities):
            print("Checking for match: ", ibd_gran_tax)
            for hmp_tax in hmp_taxonomies:
                hmp_regex = "__([^\.]*?)\."
                hmp_tax_granularities = re.findall(hmp_regex, hmp_tax)
                if ibd_gran_tax in hmp_tax_granularities:
                    matched_tax.append([ibd_tax, hmp_tax])
                    print("Found match: ", ibd_tax, hmp_tax, " moving onto next taxonomy")
                    found_match = True
                if found_match:
                    break
            if found_match:
                break
    matched_tax = pd.DataFrame(matched_tax, columns=["IBD Taxonomy", "HMP Taxonomy"])
    matched_tax.to_csv("./data/matched_taxonomies.csv")
    # Filter out the common taxonomies
    hmp_data = hmp_data[hmp_data.columns[hmp_data.columns.isin(matched_tax[:][1])]]
    return hmp_data

def hmp_tax_parsing(hmp_taxonomies, ibd_taxonomies):
    # Find common taxonomies between HMP and IBDMDB datasets using fuzzy matching
    matched_tax = []
    ibd_taxonomies = list(ibd_taxonomies["taxonomy"])
    for hmp_tax in hmp_taxonomies:
        hmp_regex = "__([^\.]*?)\."
        hmp_tax_granularities = re.findall(hmp_regex, hmp_tax)
        found_match = False
        print(hmp_tax_granularities, found_match)
        for hmp_gran_tax in reversed(hmp_tax_granularities):
            print("Checking for match: ", hmp_gran_tax)
            for ibd_tax in ibd_taxonomies:
                ibd_regex = "__([^;]*?)(?:;|$)"
                ibd_tax_granularities = re.findall(ibd_regex,ibd_tax)
                if hmp_gran_tax in ibd_tax_granularities:
                    matched_tax.append([ibd_tax, hmp_tax])
                    print("Found match: ", ibd_tax, hmp_tax, " moving onto next taxonomy")
                    found_match = True
                if found_match:
                    break
            if found_match:
                break
    matched_tax = pd.DataFrame(matched_tax, columns=["IBD Taxonomy", "HMP Taxonomy"])
    matched_tax.to_csv("./data/matched_taxonomies_hmp.csv", index=False)
    return matched_tax

def combine_datasets(hmp_data, ibd_data, matched_tax, ibd_taxonomies):
    # Combine the datasets
    columns_to_keep = matched_tax["IBD Taxonomy"].unique()
    lookup_list = matched_tax.groupby("IBD Taxonomy")["HMP Taxonomy"].apply(list)
    column_mapping = ibd_taxonomies.set_index("taxonomy")["#OTU ID"].to_dict()
    for i,_ in enumerate(columns_to_keep):
        columns_to_keep[i] = column_mapping[columns_to_keep[i]]
    np.append(columns_to_keep,"Label")
    ibd_data = ibd_data[columns_to_keep]
    print(ibd_data)
    ibd_data.to_csv("./data/ibdmdb/ibd_pathabundance_final.csv")
    renamed_list = lookup_list.rename(column_mapping)
    print(len(renamed_list))
    aggregated_values = {}
    aggregated_values["#OTU ID"] = hmp_data["SRS"]
    for new_row_name, columns_to_aggregate in renamed_list.items():
        aggregated_values[new_row_name] = hmp_data[columns_to_aggregate].astype(float).sum(axis=1)
    aggregated_values = pd.DataFrame(aggregated_values)
    aggregated_values.set_index("#OTU ID", inplace=True)
    # Apply sofmax here
    aggregated_values["Label"] = "Healthy"
    aggregated_values.to_csv("./data/hmp/hmp_stool_dataset.csv")
    print(aggregated_values)
    

    return ibd_data


# hmp_data, hmp_taxonomies = load_hmp("./data/hmp/healthy_pathabundance.csv")
# hmp_data = load_bulk_hmp("hmp1-II_humann2_pathabundance-nrm-mtd-qcd.tsv")

# hmp_data = pd.read_csv("./data/hmp/healthy_pathabundance_genus.csv", index_col=0)
# hmp_data = hmp_data.transpose()
# hmp_data.set_index("SRS", inplace=True)
# hmp_data.drop(columns=["Unnamed: 0", "Label", "RANDSID", "VISNO", "STArea", "STSite","SNPRNT","Gender","WMSPhase",], axis=1,inplace=True)
# print(hmp_data)
# print(hmp_data.columns)
# hmp_data = hmp_data.astype(float)
# hmp_data = hmp_data.groupby(hmp_data.columns, axis=1).sum()
# hmp_data.to_csv("./data/hmp/healthy_pathabundance_genus_summed.csv")
# print(hmp_data)

# ibd_data, ibd_taxonomies = load_ibd("./data/ibdmdb/taxonomic_profiles.tsv")

# matched_tax = hmp_tax_parsing(hmp_taxonomies, ibd_taxonomies)
# matched_tax = pd.read_csv("./data/matched_taxonomies.csv")

# final_data = combine_datasets(hmp_data, ibd_data, matched_tax, ibd_taxonomies)
# Graphing the IBDMDB dataset to see the distribution of the labelled data

# collapse_genus(hmp_data)


# sns.set_theme(style="whitegrid")
# plt.figure(figsize=(10, 6))
# # sns.countplot(x=ibd_data["Label"])
# plt.title("IBDMDB Dataset Distribution")
# plt.xlabel("Label")
# plt.ylabel("Count")
# plt.show()

hmp_oldest = pd.read_csv("./data/hmp/healthy_pathabundance.csv", index_col=0)
hmp_ibdonedata = pd.read_csv("./data/hmp/normalized_hmp_data.csv", index_col=0)
hmp_ibdtwodata = pd.read_csv("./data/hmp/normalized_hmp_final.csv", index_col=0)

ibdone = pd.read_csv("data/ibdmdb/taxonomic_profiles.tsv", sep="\t", index_col=0)
ibdoneoriginal = pd.read_csv("./data/ibdmdb/normalized_ibd_data.csv", index_col=0)

ibdtwo = pd.read_csv("./data/ibdmdb/qiime2_otu_with_taxonomy_original.csv", index_col=0)
ibdtwodata = pd.read_csv("./data/ibdmdb/normalized_ibd_final.csv", index_col=0)

print("HMP Original Shape:", hmp_oldest.shape)
print("HMP Data Shape: ", hmp_ibdonedata.shape)
print("HMP Final Shape: ", hmp_ibdtwodata.shape)
print("IBD One Shape: ", ibdone.shape)
print("IBD One Data Shape: ", ibdoneoriginal.shape)
print("IBD Two Shape: ", ibdtwo.shape)
print("IBD Two Data Shape: ", ibdtwodata.shape)