import configparser
import argparse
import logging
from pathlib import Path

# path to default config file
PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULTS = PROJECT_DIR / 'source_code' / 'default_config.cfg'


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
    conf_beta = float(config['PARAMS']['BETA'])
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
    fin_beta = pargs_beta if pargs_beta else conf_beta
    fin_k = pargs_k if pargs_k else conf_k
    fin_iterations = pargs_iterations if pargs_iterations else conf_iterations
    fin_runs = pargs_runs if pargs_runs else conf_runs
    fin_clusters = pargs_clusters if pargs_corpus_long else conf_clusters
    fin_num_words = pargs_num_words if pargs_num_words else conf_num_words

    # setup logging
    log_filename = str(PROJECT_DIR / fin_log_dir / 'run_gsdmm.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_filename, level=logging.INFO)

    logger.info('*****LOADING FILES AND CHECKING FILE PATHS******')
    short_corpus_filename = PROJECT_DIR / fin_data_dir / fin_corpus_short
    long_corpus_filename = PROJECT_DIR / fin_data_dir / fin_corpus_long
    true_label_filename = PROJECT_DIR / fin_data_dir / fin_true_labels
    predicted_label_pickles = PROJECT_DIR / fin_pickle_dir


if __name__ == '__main__':
    main()
