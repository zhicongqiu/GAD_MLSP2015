# GAD_MLSP2015

main file is in get_top_anomaly

goal: extract group anomalies from DATA

ask author for source file

###################################################
Variables:

TRAIN:

1) a .txt file, 2-d array with (n_sample,n_feature)

2) used to train the background model (GMMs, MIs)

DATA:

1) a .txt file, 2-d array with (n_sample,n_feature)

2) extract group anomalies from DATA

LABEL:

1) each DATA sample's label

normal_cat:

indicate the normal label

M_max:

max number of components each gmm can attain