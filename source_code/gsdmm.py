import numpy as np
from collections import Counter
from tqdm import tqdm
import preprocess


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
        self.word_count_num_topics_by_vocab_size = np.zeros((num_topics, vocab_size), dtype=np.uintc)
        self.topic_assignment_num_docs_by_num_topics = np.zeros((self.num_docs, num_topics), dtype=np.uintc)
        # TODO change topic_assignment_num)docs_by_num_topic to just the shape of num_doc, simply store the topic
        #  index at the location of doc_index

        """
        Initialization step: topic assignment
        
        Randomly assign a topic, P(topic) ~ Multinomial(1/num_topics)
        Increment number of documents for the assigned topic
        Add number of words in the doc to total number of words in the assigned topic
        Increment number of words for the assigned topic for each word in vocab (word frequency distribution by topic)
        
        topic_assignment array is a 0-filled array of shape num_docs x num_topics
        The topic_index location where the entry is NOT 0 is the topic label;
        For the example array below, doc_index 0 is assigned topic label 2 (out of 5 topic labels):
        array([[0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        """
        for doc_index, each_doc in enumerate(documents):
            topic = np.random.multinomial(1, (1 / float(num_topics)) * np.ones(num_topics))
            topic_index = topic.argmax()
            num_words_in_doc = len(each_doc)
            self.topic_assignment_num_docs_by_num_topics[doc_index, :] = topic
            self.num_docs_per_topic[topic_index] += 1
            self.num_words_per_topic[topic_index] += num_words_in_doc
            for word_id in each_doc:
                self.word_count_num_topics_by_vocab_size[topic_index, word_id] += 1

    def topic_reassignment(self):
        # GSDMM algorithm in each iteration
        for doc_index, each_doc in enumerate(tqdm(self.documents)):
            # record current topic assignment from the initialization step
            # exclude doc and word frequencies for this topic
            current_topic_index = self.topic_assignment_num_docs_by_num_topics[doc_index, :].argmax()
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
            self.topic_assignment_num_docs_by_num_topics[doc_index, :] = new_topic
            self.num_docs_per_topic[new_topic_index] += 1
            self.num_words_per_topic[new_topic_index] += num_words_in_doc
            for word_id in each_doc:
                self.word_count_num_topics_by_vocab_size[new_topic_index, word_id] += 1

        return None

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

        log(right numerator) == sum(sum(log(word_count_topics_by_vocab[topic, word_id] + beta + j -1)))
        log(right denominatory) == sum(log(num_words_per_topic[topic_index] + vocab_size * beta + i - 1))


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

    def iterate_topic_reassignment(self, iterations=15):
        for iteration in range(iterations):
            self.topic_reassignment()

        return None

    def document_best_topic_label(self):
        for doc_index in range(self.num_docs):
            topic_label = self.topic_assignment_num_docs_by_num_topics[doc_index, :].argmax()
            print(f'Doc no: {doc_index} is assigned topic label: {topic_label}')

        return None


def most_populated_clusters(gsdmm, vocab):
    highest_num_docs = list(np.sort(gsdmm.num_docs_per_topic)[::-1])[:10]
    most_docs_topics = list(np.argsort(gsdmm.num_docs_per_topic)[::-1])[:10]
    print(f'Number of documents per topic for most populated clusters: {highest_num_docs}')
    print(f'Topic labels with highest numbers of documents: {most_docs_topics}')

    for topic in most_docs_topics:
        most_freq_words_ids = list(np.argsort(gsdmm.word_count_num_topics_by_vocab_size[topic, :])[::-1])[:5]
        highest_word_freq = list(np.sort(gsdmm.word_count_num_topics_by_vocab_size[topic, :])[::-1])[:5]
        most_frequent_words = [(vocab.id_to_word[word_id], freq) for word_id, freq in zip(most_freq_words_ids,
                                                                                          highest_word_freq)]
        print(f'Topic label: {topic}\tMost frequent words: {most_frequent_words}')

    return None


def main():
    toy_filename = '../data/toy.txt'
    toy_corpus = preprocess.load_corpus(toy_filename)

    vocab = preprocess.Vocabulary()
    docs = [vocab.doc_to_ids(doc) for doc in toy_corpus]

    gsdmm = GSDMM(docs, vocab.size())

    gsdmm.iterate_topic_reassignment()
    gsdmm.document_best_topic_label()

    most_populated_clusters(gsdmm, vocab)


if __name__ == '__main__':
    main()
