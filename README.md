# Gibbs Samping Dirichlet Multinomial Model
CompLing Final Project GSDMM Implementation as outlined in Yin and Wang (2014)
To Do's
- 2 data sets, 1 short texts and 1 long text
    - Pang and Lee (2004) Movie review dataset (only to show topic modeling)
    - 2015NAACL VSM-NLP workshop-"Short Text Clustering via Convolutional Neural Networks" dataset
- ~~pro-process corpora the same way; use the same pro-processing procedures as in the papers (Mazora; Yin amnd Wang)~~
- ~~Create a toy corpus for testing~~
- ~~Initialize~~
- ~~Write function to calculate the posterior probability for sampling new topic~~
- ~~Show words in documents of the final clusters~~
- Show results/evaluate per metrics in the Y and W paper
- write eval script to tabulate: H, C, NMI, ARI, AMI per Yin and Wang paper
- if time allows, visualization with pyLDA??? 
- keep track of average doc len, num of non-0 clusters per iteration
- to pickle: predicted labels, dict of true/predicted most frequent words/topic
- numpy.save/load: 
- write argparse arguments for options and filenames