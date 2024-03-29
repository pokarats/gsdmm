# Gibbs Samping Dirichlet Multinomial Mixture Model (GSDMM) in Short-Text Clustering

Computational Linguistics Final Project, Winter Semester 2019, University of Saarland  
GSDMM Implementation as described in:  
_**Yin, J. and Wang., J. A dirichlet multinomial mixturemodel-based approach for short text clustering. In SIGKDD,2014**_  
Experimenting with different beta parameters on the Stack Overflow Titles dataset made available by Kaggle.com from the paper by:  
_**Xu, J., et al., 2015. Short Text Clustering via Convolutional Neural Networks,
NAACL.**_  

## Project File Structure

- GSDMM
    - README.md
    - gsdmm_noonPokaratsiriGoldstein.pdf _project report_
    - data: _all corpus and label files are here_   
        - title_StackOverflow.txt
        - label_StackOverflow.txt
    - logs: _logs of run_gsdmm.py execution_
        - run_gsdmm_{run_id}.log
    - output: _plots of gsdmm performance and representative words in clusters_
        - cluster_per_iteration_at_different_beta.png
        - performance_at_different_beta.png
        - gsdmm_clusters_and_representative_words_{run_id}.out
    - pickled: _pickle files from run_gsdmm.py_
        - predicted_{run_id}_freq_words_by_beta.pickle
        - predicted_{run_id}_labels_by_beta.pickle
        - predicted_{run_id}_num_clusters_by_it_per_beta_list.pickle
        - true_most_frequent_words_by_topic.pickle
    - source_code: _config file for default parameters and all source code files_  
    **execute run_gsdmm.py from this directory**
        - default_config.cfg _default parameters to execute the program are defined here_
        - eval.py _this module calculates NMI, Homogeneity, and Completeness and plot graphs_
        - gsdmm.py _this module does the GSDMM algorithm_
        - preprocess.py _this module tokenizes and pre process the corpus file_
        - run_gsdmm.py _**this is the main program that runs the experiment**_

## Requirements

Python 3.7

numpy  
sklearn  
matplotlib  
nltk  
tqdm

## Instructions

- cd to the source_code directory to execute the program
- **python run_gsdmm.py -h** will display all the command line options
- commandline options will override options in the **default_config.cfg** file
- **python run_gsdmm.py** will run GSDMM experiments with the default values in the .cfg file
- the last run_id was 3; change to a different run_id number to execute the full program
- program will output 2 plots (plot titles are self-explanatory), an output file showing the GSDMM predicted number of clusters, words in the clusters + frequencies
- running the program with the same run_id will simply load data from pickled files and re-plot the 2 graphs
- runtime: for K = 100 (starting with 100 clusters as an upper bound), the program takes approximately **1 hour for each beta value computation** in the experiment. For K = 50, each cycle takes approximately 30 minutes. 
- The default setting experiments with **5 beta values**; therefore, the **total runtime** for the entire program takes **approximately 5-6 hours**.
- Please see the log file for runtime details as it includes time stamps from the last run