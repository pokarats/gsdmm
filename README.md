# Gibbs Samping Dirichlet Multinomial Model (GSDMM)
Computational Linguisitics Final Project GSDMM Implementation as outlined in Yin and Wang (2014)

Project File Structure:

- GSDMM
    README.md
    - data: _all corpus and label files are here_
        - title_StackOverflow.txt
        - label_StackOverflow.txt
    - logs: _logs of run_gsdmm.py execution_
        - run_gsdmm_{run_id}.log
    - output: _plots of gsdmm performance and representative words in clusters_
        - cluster_per_iteration_at_different_beta.png
        - performance_at_different_beta.png
        - gsdmm_clusters_and_representative_words.out
        - gsdmm_clusters_and_representative_words_{run_id}.out
    - pickled: _pickle files from run_gsdmm.py_
        - predicted_{run_id}_freq_words_by_beta.pickle
        - predicted_{run_id}_labels_by_beta.pickle
        - predicted_{run_id}_num_clusters_by_it_per_beta_list.pickle
        - true_most_frequent_words_by_topic.pickle
    - source_code: _config file for default parameters and all source code files_
        - default_config.cfg
        - eval.py
        - gsdmm.py
        - preprocess.py
        - run_gsdmm.py

Requirements:

Python 3.7

numpy
sklearn
matplotlib
nltk
tqdm

Instructions:

- 