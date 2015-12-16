# GAD_MLSP2015

MAIN FILE:

get_top_anomaly.py

GOAL: 

extract prioritized clusters of anomalous samples and their salient (continuous, real-valued) feature subsets from DATA

Based on our MLSP paper:

Detecting Clusters of Anomalies on Low-Dimensional Feature Subsets with
Application to Network Traffic Flow Data

In that paper, public-domain TCPdump traces of background web enterprise network packet traffic and botnet command and control traffic that also uses port 80 (two different domains) were processed by wireshark to create netflows/sessions from which the bidirectional sequence of packet sizes were extracted to form the DATA used in this case study. The anomaly detection framework is fully unsupervised in that the botnet activity to be detected is only resident in the (unlabelled) test set, together with a fraction of the (nominal) background traffic samples (multifold cross validation was performed). We therefore did not use the categorical features from the headers. Also, the packet payloads were unavailable (public-domain datasets), and the timing features were not used (because the background and botnet traces were recorded in different domains). 

Paper Links: 

arvix: 

http://arxiv.org/pdf/1511.01047.pdf

MLSP'15 IEEE Xplore:  

http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7324326&tag=1


Also see: actively learning to distinguish suspicious from innocuous anomalies in a batch of vehicle tracks

Active learning paper in spie for 2 classes to distinguish suspicious from innocuous anomalies.

Paper Link:

http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1881997

Pending journal paper that describes a both active and semi-supervised framework for >2 (generally novel/rare) classes.

###################################################
-----------------------------------------------------
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
