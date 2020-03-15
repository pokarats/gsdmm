import timeit
import logging
import numpy as np
from collections import Counter
from tqdm import tqdm, trange
import preprocess
import pickle


class GSDMM:
    def __init__(self, documents, vocab_size, num_topics=50, alpha=0.1, beta=0.1):
        self.documents = documents
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.beta = beta
        self.num_docs = len(self.documents)

        self.num_docs_per_topic = np.zeros(num_topics, dtype=np.uintc)
        self.num_words_per_topic = np.zeros(num_topics, dtype=np.uintc)
        self.topic_label_by_doc = np.zeros(self.num_docs, dtype=np.uintc)
        self.word_count_num_topics_by_vocab_size = np.zeros((num_topics, vocab_size), dtype=np.uintc)
        # self.topic_assignment_num_docs_by_num_topics = np.zeros((self.num_docs, num_topics), dtype=np.uintc)

        """
        Initialization step: topic assignment
        
        Randomly assign a topic, P(topic) ~ Multinomial(1/num_topics)
        Increment number of documents for the assigned topic
        Add number of words in the doc to total number of words in the assigned topic
        Increment number of words for the assigned topic for each word in vocab (word frequency distribution by topic)
        
        np.random.multinomial(1, [(1/num_topics) * num_topics]) returns an array of len(num_topics) where the non-0
        index location is the randomly sampled choice, i.e. the assigned topic
        """
        for doc_index, each_doc in enumerate(documents):
            topic_index = np.random.multinomial(1, (1 / float(num_topics)) * np.ones(num_topics)).argmax()
            num_words_in_doc = len(each_doc)
            self.topic_label_by_doc[doc_index] = topic_index
            # self.topic_assignment_num_docs_by_num_topics[doc_index, :] = topic
            self.num_docs_per_topic[topic_index] += 1
            self.num_words_per_topic[topic_index] += num_words_in_doc
            for word_id in each_doc:
                self.word_count_num_topics_by_vocab_size[topic_index, word_id] += 1

    def _count_non_zero_docs_topics(self):
        return len(self.num_docs_per_topic.nonzero()[0])

    def gibbs_sampling_topic_reassignment(self, iterations=10):
        num_non_zero_topic_clusters = []
        for _ in trange(iterations):
            # keeping track of number of topics with docs in each iteration
            num_non_zero_topic_clusters.append(self._count_non_zero_docs_topics())
            # GSDMM algorithm in one iteration
            for doc_index, each_doc in enumerate(self.documents):
                # record current topic assignment from the initialization step
                # exclude doc and word frequencies for this topic
                current_topic_index = self.topic_label_by_doc[doc_index]
                # current_topic_index = self.topic_assignment_num_docs_by_num_topics[doc_index, :].argmax()
                num_words_in_doc = len(each_doc)
                self.num_docs_per_topic[current_topic_index] -= 1
                self.num_words_per_topic[current_topic_index] -= num_words_in_doc
                for word_id in each_doc:
                    self.word_count_num_topics_by_vocab_size[current_topic_index, word_id] -= 1

                # re-sample for a new topic assignment based on Equation 4 in Yin and Wang
                prob_topic_assigned_to_doc = self.calc_normalized_topic_sampling_prob(each_doc)
                # print(prob_topic_assigned_to_doc)
                new_topic = np.random.multinomial(1, prob_topic_assigned_to_doc)
                new_topic_index = new_topic.argmax()

                # update doc and word counts based on new topic assignment
                self.topic_label_by_doc[doc_index] = new_topic_index
                # self.topic_assignment_num_docs_by_num_topics[doc_index, :] = new_topic
                self.num_docs_per_topic[new_topic_index] += 1
                self.num_words_per_topic[new_topic_index] += num_words_in_doc
                for word_id in each_doc:
                    self.word_count_num_topics_by_vocab_size[new_topic_index, word_id] += 1

        return num_non_zero_topic_clusters

    def calc_normalized_topic_sampling_prob(self, doc):
        """
        Equation 4 from Yin and Wang (2014) represents the probability of a document being assigned a topic K
        i.e. prob[topic_index]

        Breaking up Equation 4 into 4 components:  left numerator * right numerator
                                                   ------------------------------------
                                                   left denominator * right denominator

        left numerator: num_docs_per_topic[topic_index] + alpha
        left denominator: num docs in corpus - 1 + num_topics * alpha

        right numerator: product(product(word_count_topics_by_vocab[topic, word_id] + beta + j - 1))
        from j == 1 to word frequency of word w in doc, for each word_id in doc

        right denominator: product(num_words_per_topic[topic_index] + vocab_size * beta + i - 1) from i == 1 to
        num_words in doc

        Working in natural log space to avoid underflow:

        Equation 4 == exp(log(left numerator) + log(right numerator) - log(left denominator) - log(right denominator))

        log(left numerator) == log(num_docs_per_topic[topic_index] + alpha)
        log(left denominator) == log(num docs in corpus - 1 + num_topics * alpha)

        log(right numerator) == sum(sum(log(word_count_topics_by_vocab[topic, word_id] + beta + j - 1)))
        log(right denominator) == sum(log(num_words_per_topic[topic_index] + vocab_size * beta + i - 1))


        :param doc: tokenized doc, each word token is an integer representation
        :type doc: List[int]
        :return: np.array of normalized probabilities for a doc being assigned each topic for all topics
        """
        ln_prob = np.zeros(self.num_topics)
        doc_word_freq = Counter(doc)
        num_words_in_doc = len(doc)

        # calculating probability for each topic_index in natural log space
        ln_left_denominator = np.log(self.num_docs - 1 + self.num_topics * self.alpha)
        for topic_index in range(self.num_topics):
            ln_left_numerator = np.log(self.num_docs_per_topic[topic_index] + self.alpha)
            ln_right_numerator = 0.0
            ln_right_denominator = 0.0
            for word_id in doc:
                word_freq = doc_word_freq[word_id]
                for j in range(1, word_freq + 1):
                    ln_right_numerator += np.log(self.word_count_num_topics_by_vocab_size[topic_index, word_id] +
                                                 self.beta + j - 1)
            for i in range(1, num_words_in_doc + 1):
                ln_right_denominator += np.log(self.num_words_per_topic[topic_index] + self.vocab_size * self.beta + i
                                               - 1)
            ln_prob[topic_index] = ln_left_numerator + ln_right_numerator - ln_left_denominator - ln_right_denominator

        # converting log probabilities back to linear scale
        # use 128-bit float to avoid NaN overflow
        prob = np.exp(ln_prob, dtype=np.float128)

        # normalize probabilities
        try:
            normalized_prob = prob / prob.sum()
        except ZeroDivisionError:
            normalized_prob = prob / 1.0

        # return as float64 to be compatible with np.random.multinomial
        return normalized_prob.astype(np.float64)

    def predict_doc_topic_labels(self):
        predicted_labels = []
        for doc_index in range(self.num_docs):
            topic_label = self.topic_label_by_doc[doc_index]
            predicted_labels.append(topic_label)
            # topic_label = self.topic_assignment_num_docs_by_num_topics[doc_index, :].argmax()
            # print(f'Doc no: {doc_index} is assigned topic label: {topic_label}')

        return predicted_labels


def make_pickle(filename, obj_to_pickle):
    logging.getLogger(__name__).info(f'dumping pickle file to: {filename}')
    with open(filename, 'wb') as w_file:
        pickle.dump(obj_to_pickle, w_file)
    return None


def predict_most_populated_clusters(gsdmm, vocab, filename, num_wanted_words=5, num_wanted_topics=20):
    highest_num_docs = np.sort(gsdmm.num_docs_per_topic)[::-1][:num_wanted_topics]
    most_docs_topics = np.argsort(gsdmm.num_docs_per_topic)[::-1][:num_wanted_topics]
    with open(filename, 'a') as w_file:
        print(f'Predicted number of documents per topic for most populated clusters: {highest_num_docs}\n'
              f'Predicted topic labels with highest numbers of documents: {most_docs_topics}', file=w_file)

    most_frequent_words_by_topic = {}
    with open(filename, 'a') as w_file:
        for topic in most_docs_topics:
            most_freq_words_ids = np.argsort(gsdmm.word_count_num_topics_by_vocab_size[topic, :])[::-1][:num_wanted_words]
            highest_word_freq = np.sort(gsdmm.word_count_num_topics_by_vocab_size[topic, :])[::-1][:num_wanted_words]
            most_frequent_words = [(vocab.id_to_word[word_id], freq) for word_id, freq in zip(most_freq_words_ids,
                                                                                              highest_word_freq)]
            most_frequent_words_by_topic[topic] = most_frequent_words
            print(f'Predicted topic label: {topic}\tMost frequent words: {most_frequent_words}', file=w_file)

    return most_frequent_words_by_topic


def true_most_populated_clusters(true_clusters, documents, vocab, filename, num_wanted_words=5, num_wanted_topics=20):
    logging.getLogger(__name__).info(f'Starting output file with true clusters, saving to: {filename}')
    # true_clusters is a list of list of docs in a topic, len(list of docs) == num_docs_per_topic
    num_topics = len(true_clusters)
    cluster_size = []
    for cluster in true_clusters:
        cluster_size.append(len(cluster))

    num_docs_per_topic = cluster_size[:num_wanted_topics]

    with open(filename, 'w') as w_file:
        print(f'Number of documents per topic in true clusters: {num_docs_per_topic}', file=w_file)

    word_count_per_topic = np.zeros(num_topics, dtype=np.uintc)
    word_count_num_topics_by_vocab_size = np.zeros((num_topics, vocab.size()), dtype=np.uintc)
    for topic_index, each_topic in enumerate(true_clusters):
        for each_doc_id in each_topic:
            word_count_per_topic[topic_index] += len(documents[each_doc_id])
            for word_id in documents[each_doc_id]:
                word_count_num_topics_by_vocab_size[topic_index, word_id] += 1

    most_frequent_words_by_topic = {}
    with open(filename, 'a') as w_file:
        for topic_index in range(num_topics):
            most_freq_words_ids = np.argsort(word_count_num_topics_by_vocab_size[topic_index, :])[::-1][
                                  :num_wanted_words]
            highest_word_freq = np.sort(word_count_num_topics_by_vocab_size[topic_index, :])[::-1][:num_wanted_words]
            most_frequent_words = [(vocab.id_to_word[word_id], freq) for word_id, freq in zip(most_freq_words_ids,
                                                                                              highest_word_freq)]
            most_frequent_words_by_topic[topic_index] = most_frequent_words
            print(f'True topic label: {topic_index}\tMost frequent words: {most_frequent_words}', file=w_file)

    return most_frequent_words_by_topic


def main():
    toy_filename = '../data/toy.txt'
    sofl_filename = '../data/title_StackOverflow.txt'
    toy_corpus = preprocess.load_corpus(toy_filename)
    stack_overflw_corpus = preprocess.load_corpus(sofl_filename)

    toy_vocab = preprocess.Vocabulary()
    toy_docs = [toy_vocab.doc_to_ids(doc) for doc in toy_corpus]

    vocab = preprocess.Vocabulary()
    stack_overflow_docs = [vocab.doc_to_ids(doc) for doc in stack_overflw_corpus]

    gsdmm_toy = GSDMM(toy_docs, toy_vocab.size())
    num_toy_topic_clusters_by_iterations = gsdmm_toy.gibbs_sampling_topic_reassignment()
    toy_predicted_labels = gsdmm_toy.predict_doc_topic_labels()
    toy_predicted_most_freq_words_by_topic = predict_most_populated_clusters(gsdmm_toy, toy_vocab)

    # true labels and clusters
    labels = preprocess.load_labels('../data/label_StackOverflow.txt')
    true_clusters = preprocess.make_topic_clusters(labels)
    true_most_frequent_words_by_topic = true_most_populated_clusters(true_clusters, stack_overflow_docs, vocab)

    #full stack_overflow run
    start_timer = timeit.default_timer()
    stack_gsdmm = GSDMM(stack_overflow_docs, vocab.size())
    num_stack_topics_by_iter = stack_gsdmm.gibbs_sampling_topic_reassignment()
    stack_predicted_labels = stack_gsdmm.predict_doc_topic_labels()
    print(f'number of non-0 topics by iteration: {num_stack_topics_by_iter}')
    elapsed_time = timeit.default_timer() - start_timer
    print(f'time to run GSDMM 1: {elapsed_time}')

    # pickle predicted labels
    pickled_predicted_labels = '../pickled/predicted_labels1.pickle'
    with open(pickled_predicted_labels, 'wb') as w_file:
        pickle.dump(stack_predicted_labels, w_file)

    with open(pickled_predicted_labels, 'rb') as saved_pickle:
        loaded_predicted_labels = pickle.load(saved_pickle)

    assert stack_predicted_labels == loaded_predicted_labels
    predicted_most_freq_words_by_topic = predict_most_populated_clusters(stack_gsdmm, vocab)

    #full_stack run2
    stack2_gsdmm = GSDMM(stack_overflow_docs, vocab.size())
    num2_stack_topics_by_iter = stack2_gsdmm.gibbs_sampling_topic_reassignment()
    stack2_predicted_labels = stack2_gsdmm.predict_doc_topic_labels()
    print(f'number of non-0 topics by iteration run 2: {num2_stack_topics_by_iter}')
    make_pickle('../pickled/predicted_labels2.pickle', stack2_predicted_labels)


if __name__ == '__main__':
    main()
