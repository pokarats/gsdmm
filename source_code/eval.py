import pickle
import preprocess
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari
from sklearn.metrics import homogeneity_completeness_v_measure as hcv


def main():

    pickled_predicted_labels = '../data/predicted_labels.pickle'
    with open(pickled_predicted_labels, 'rb') as saved_pickle:
        loaded_predicted_labels = pickle.load(saved_pickle)

    # true labels and clusters
    labels = preprocess.load_labels('../data/label_StackOverflow.txt')

    assert len(labels) == len(loaded_predicted_labels)
    nmi_score = nmi(labels, loaded_predicted_labels)
    ami_score = ami(labels, loaded_predicted_labels)
    ari_score = ari(labels, loaded_predicted_labels)
    h_score, c_score, v_measure = hcv(labels, loaded_predicted_labels)

    print(f'Evaluation metrics:\n'
          f'Normalized Mutual Info: {nmi_score}\n'
          f'Adjusted Mutual Info: {ami_score}\n'
          f'Adjusted Rand Score: {ari_score}\n'
          f'Homogeneity: {h_score}\n'
          f'Completeness: {c_score}\n'
          f'V-measure: {v_measure}\n')


if __name__ == '__main__':
    main()