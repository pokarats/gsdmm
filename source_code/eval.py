import pickle
import logging
import preprocess
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
from sklearn.metrics import homogeneity_completeness_v_measure as hcv


def read_pickle(filename):
    logging.getLogger(__name__).info(f'loading pickle file: {filename}')
    with open(filename, 'rb') as saved_pickle:
        return pickle.load(saved_pickle)


def model_performance(true_labels, predicted_label_lists):
    """

    :param true_labels:
    :param predicted_label_lists:
    :return: tuple of 3 lists: nmi_scores, h_scores, c_scores
    """
    logging.getLogger(__name__).info('calculating NMI, Completeness, and Homogeneity for different betas')
    nmi_scores = []
    h_scores = []
    c_scores = []

    for each_list in predicted_label_lists:
        assert len(true_labels) == len(each_list)
        nmi_scores.append(nmi(true_labels, each_list))
        c, h, _ = hcv(true_labels, each_list)
        c_scores.append(c)
        h_scores.append(h)
    return nmi_scores, h_scores, c_scores


def plot_results(x_values, *args, x_label, y_label, title, file_directory, labels=None):
    logging.getLogger(__name__).info(f'plotting {title}')
    # only one container/list passed into the function
    # e.g. plotting num cluster / iteration
    if len(args) == 0:
        x_values = []
        y_values = []
        for idx, val in enumerate(x_values):
            x_values.append(idx)
            y_values.append(val)
        plt.plot(x_values, y_values, 'o--')
    # only 1 x-values list and 1 y-values list
    # e.g. plotting num cluster and beta values
    elif len(args) == 1:
        assert len(x_values) == len(args[0])
        plt.plot(x_values, args[0], 'o--')
    else:
        # multiple y-values lists
        # e.g. plotting NMI, C, H etc. against beta values
        if labels is not None:
            assert len(args) == len(labels)
            for group_idx, y_val_list in enumerate(args):
                plt.plot(x_values, y_val_list, 'o--', label=f'{labels[group_idx]}')
                plt.legend(loc='upper right')
        else:
            for y_val_list in args:
                plt.plot(x_values, y_val_list, 'o--')

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    filename = f'{str(file_directory)}/{title}.png'
    logging.getLogger(__name__).info(f'saving {title} plot to {filename}')
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

    return None


def main():
    # can ignore sections below as they're there for simply for testing

    test_lists = [[1,2,3], [3,4,5], [4,5,6]]
    plot_results([i for i in range(1, 4)], *test_lists, x_label='iterations', y_label='number of non-zero clusters',
                 title='test_clusters_per_iteration', file_directory='output', labels=['0.02', '0.1', '0.2'])


if __name__ == '__main__':
    main()
