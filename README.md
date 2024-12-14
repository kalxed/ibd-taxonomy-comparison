# ibd-taxonomy-comparison
Repository companion for BCB 501 project paper by Kai Davidson, David Guevara, and Kasish Gupta.

# Downloading IBDMDB Files

Navigate to ibdmdb.org and download the aggregated taxonomic abundance as a tsv file from the files tab.

# Downloading HMP Files

To download the HMP files, you must access the AWS bucket they are stored on. The correct command to download the specific file used in this project is:

`aws s3 ls s3://hmpdcc/hmp1/hhs/microbiome/wms/analysis/metabolic_prof/humann2_2017/hmp1-II_humann2_pathabundance-nrm-mtd-qcd.pcl.bz2 . --no-sign-request`

Afterwards, use the csv file in the hmp_manifests folder to filter out only healthy patients using the `create_dataset.py` functions within the file.

# Using the Models

To use the models specify the correct path for the normalized and aggregated genus data in the `data` folder. Then run `python <model.py>` to run the model and generate plots. Change the dataset within the model files to run on different datasets.