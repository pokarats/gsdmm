import configparser
import argparse
import logging
from pathlib import Path
import preprocess
import gsdmm
import eval


# path to default config file
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULTS = PROJECT_DIR / 'source_code' / 'default_config.cfg'


def experiment_with_beta(beta_list, iterations, docs, vocab, num_topics, alpha, filename, num_words, wanted_topics):
    """

    :param wanted_topics:
    :param num_words:
    :param filename:
    :param beta_list:
    :param iterations:
    :param docs:
    :param vocab:
    :param num_topics:
    :param alpha:
    :return: tuple of 3 lists: list of lists of num_cluisters per iteration, list of lists of predicted labels,
    list of lists predicted_most_frequent_words_by_topic
    """
    num_clusters_by_beta = []  # list of lists of num_clusters per iteration
    topic_labels_by_beta = []  # list of lists of predicted labels from each run
    predicted_most_freq_words_by_topic_lists = []
    for beta in beta_list:
        model = gsdmm.GSDMM(docs, vocab.size(), num_topics, alpha, beta)

        num_clusters_per_iter = model.gibbs_sampling_topic_reassignment(iterations)
        num_clusters_by_beta.append(num_clusters_per_iter)
        topic_labels_by_beta.append(model.predict_doc_topic_labels())

        # number of topics == the number of non-zero clusters after the last iteration
        end_num_topics = num_clusters_per_iter[-1]
        if end_num_topics > wanted_topics:
            wanted_topics = end_num_topics    
        most_freq_words_by_topic = gsdmm.predict_most_populated_clusters(model, vocab, filename, num_words,
                                                                         wanted_topics)
        predicted_most_freq_words_by_topic_lists.append(most_freq_words_by_topic)

    return num_clusters_by_beta, topic_labels_by_beta, predicted_most_freq_words_by_topic_lists


def main():

    # set up default configurations from config file and command line arguments for overrides
    parser = argparse.ArgumentParser()
    config = configparser.ConfigParser()

    # overridable options for file management
    parser.add_argument('--data', type=str, help='name of data directory', default=None)
    parser.add_argument('--output', type=str, help='name of directory to save output files', default=None)
    parser.add_argument('--logging', type=str, help='name of log files directory', default=None)
    parser.add_argument('--pickle', type=str, help='name of pickled files directory', default=None)
    parser.add_argument('--short_corpus', type=str, help='short text corpus filename', default=None)
    parser.add_argument('--long_corpus', type=str, help='long text corpus filename', default=None)
    parser.add_argument('--labels', type=str, help='true label filename', default=None)

    # overridable options for GSDMM parameters
    parser.add_argument('--alpha', type=float, help='alpha value for Dirichlet', default=None)
    parser.add_argument('--beta', type=float, help='beta value for Dirichlet', default=None)
    parser.add_argument('--k', type=int, help='upper bound of expected number of clusters', default=None)
    parser.add_argument('--iterations', type=int, help='number of iterations to reapeat Gibbs sampling', default=None)
    parser.add_argument('--runs', type=int, help='how many times of to run GSDMM on each corpus', default=None)
    parser.add_argument('--clusters', type=int, help='number of clusters to show in output', default=None)
    parser.add_argument('--num_words', type=int, help='number of words in clusters to show in output', default=None)

    # parse command line overrides
    pargs = parser.parse_args()

    pargs_data_dir = pargs.data
    pargs_output_dir = pargs.output
    pargs_log_dir = pargs.logging
    pargs_pickle_dir = pargs.pickle
    pargs_corpus_short = pargs.short_corpus
    pargs_corpus_long = pargs.long_corpus
    pargs_true_labels = pargs.labels

    pargs_alpha = pargs.alpha
    pargs_beta = pargs.beta
    pargs_k = pargs.k
    pargs_iterations = pargs.iterations
    pargs_runs = pargs.runs
    pargs_clusters = pargs.clusters
    pargs_num_words = pargs.num_words


    # read in default values from file
    config.read(DEFAULTS)
    # default values for files and directories
    conf_data_dir = config['SETTINGS']['DATA_DIR']
    conf_output_dir = config['SETTINGS']['OUTPUT_DIR']
    conf_log_dir = config['SETTINGS']['LOG_DIR']
    conf_pickle_dir = config['SETTINGS']['PICKLE_DIR']
    conf_corpus_short = config['SETTINGS']['CORPUS_FILE_SHORT']
    conf_corpus_long = config['SETTINGS']['CORPUS_FILE_LONG']
    conf_true_labels = config['SETTINGS']['TRUE_LABELS_FILE']

    # default values for GSDMM parameters
    conf_alpha = float(config['PARAMS']['ALPHA'])
    conf_beta = [float(beta) for beta in config['PARAMS']['BETA'].split(',')]
    conf_k = int(config['PARAMS']['K'])
    conf_iterations = int(config['PARAMS']['ITERATIONS'])
    conf_runs = int(config['PARAMS']['RUNS'])
    conf_clusters = int(config['PARAMS']['CLUSTERS'])
    conf_num_words = int(config['PARAMS']['NUM_WORDS'])

    # final settings/parameters
    fin_data_dir = pargs_data_dir if pargs_data_dir else conf_data_dir
    fin_output_dir = pargs_output_dir if pargs_output_dir else conf_output_dir
    fin_log_dir = pargs_log_dir if pargs_log_dir else conf_log_dir
    fin_pickle_dir = pargs_pickle_dir if pargs_pickle_dir else conf_pickle_dir
    fin_corpus_short = pargs_corpus_short if pargs_corpus_short else conf_corpus_short
    fin_corpus_long = pargs_corpus_long if pargs_corpus_long else conf_corpus_long
    fin_true_labels = pargs_true_labels if pargs_true_labels else conf_true_labels

    fin_alpha = pargs_alpha if pargs_alpha else conf_alpha
    fin_beta = [pargs_beta] if pargs_beta else conf_beta
    fin_k = pargs_k if pargs_k else conf_k
    fin_iterations = pargs_iterations if pargs_iterations else conf_iterations
    fin_runs = pargs_runs if pargs_runs else conf_runs
    fin_clusters = pargs_clusters if pargs_corpus_long else conf_clusters
    fin_num_words = pargs_num_words if pargs_num_words else conf_num_words

    # setup logging
    log_filename = str(PROJECT_DIR / fin_log_dir / 'run_gsdmm.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s %(name)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)

    logger.info('***START GSDMM***')
    # all filenames are pathlib Path not string
    short_corpus_filename = PROJECT_DIR / fin_data_dir / fin_corpus_short
    long_corpus_filename = PROJECT_DIR / fin_data_dir / fin_corpus_long
    true_label_filename = PROJECT_DIR / fin_data_dir / fin_true_labels
    predicted_pickles = PROJECT_DIR / fin_pickle_dir / 'predicted'
    true_pickles = PROJECT_DIR / fin_pickle_dir / 'true'
    output_dir = PROJECT_DIR / fin_output_dir
    output_filename = output_dir / 'gsdmm_clusters_and_representative_words.out'

    # loading files and pre-processing
    logger.info(f'loading and preprocessing corpus files for short and long texts from'
                f'{short_corpus_filename} and {long_corpus_filename}')
    short_text_corpus = preprocess.load_corpus(short_corpus_filename)
    long_text_corpus = preprocess.load_corpus(long_corpus_filename)

    short_text_vocab = preprocess.Vocabulary()
    short_text_docs = [short_text_vocab.doc_to_ids(doc) for doc in short_text_corpus]

    long_text_vocab = preprocess.Vocabulary()
    long_text_docs = [long_text_vocab.doc_to_ids(doc) for doc in long_text_corpus]

    # loading true labels and representative words in true clusters
    logger.info(f'loading true label file from {true_label_filename}')
    true_labels = preprocess.load_labels(true_label_filename)
    true_clusters = preprocess.make_topic_clusters(true_labels)

    true_pickle_filename = f'{str(true_pickles)}_most_freq_words_by_topic.pickle'
    try:
        true_most_frequent_words_by_topic = eval.read_pickle(true_pickle_filename)
    except FileNotFoundError:
        logger.info('No pickle found')
        true_most_frequent_words_by_topic = gsdmm.true_most_populated_clusters(true_clusters, short_text_docs,
                                                                               short_text_vocab, output_filename,
                                                                               fin_num_words)
        gsdmm.make_pickle(true_pickle_filename, true_most_frequent_words_by_topic)

    # running gsdmm on different beta values or load from pickled
    logger.info(f'experimenting with betas: {fin_beta}\n'
                f'each run is for {fin_iterations} iterations'
                f'running model on short text corpus\n')
    predicted_clusters_pickle_file = f'{str(predicted_pickles)}_num_clusters_by_it_per_beta_list.pickle'
    predicted_labels_pickle_file = f'{str(predicted_pickles)}_labels_by_beta.pickle'
    predicted_freq_words_pickle_file = f'{str(predicted_pickles)}_freq_words_by_beta.pickle'
    try:
        num_clusters_by_beta = eval.read_pickle(predicted_clusters_pickle_file)
        predicted_labels_by_beta = eval.read_pickle(predicted_labels_pickle_file)
        most_freq_words_by_beta = eval.read_pickle(predicted_freq_words_pickle_file)
    except FileNotFoundError:
        # running gsdmm on different beta values
        num_clusters_by_beta, predicted_labels_by_beta, most_freq_words_by_beta = experiment_with_beta(
            fin_beta, fin_iterations, short_text_docs, short_text_vocab, fin_k, fin_alpha, output_filename,
            fin_num_words, fin_clusters)
        gsdmm.make_pickle(predicted_clusters_pickle_file, num_clusters_by_beta)
        gsdmm.make_pickle(predicted_labels_pickle_file, predicted_labels_by_beta)
        gsdmm.make_pickle(predicted_freq_words_pickle_file, most_freq_words_by_beta)

    # plotting progression of number of non-zero clusters with each iteration
    logger.info(f'plotting number of clusters per iteration with changing betas in short text corpus')
    plot_group_label = [f'beta = {str(beta)}' for beta in fin_beta]
    eval.plot_results([i for i in range(1, 11)], *num_clusters_by_beta, x_label='Iterations',
                      y_label='Number of Non-Zero Clusters', title='cluster_per_iteration_at_different_beta',
                      file_directory=output_dir, labels=plot_group_label)

    # evaluate model performance in terms of NMI, Homogeneity and Completeness
    nmi_list, h_list, c_list = eval.model_performance(true_labels, predicted_labels_by_beta)
    logger.info(f'plotting eval metrics short corpus: NMI, Homogeneity, Completeness with changing betas')
    eval.plot_results(fin_beta, nmi_list, h_list, c_list, x_label='Beta Values', y_label='Performance',
                      title='performance_at_different_beta', file_directory=output_dir, labels=
                      ['NMI, Homogeneity, Completeness'])


if __name__ == '__main__':
    main()
