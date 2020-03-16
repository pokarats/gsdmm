#!/usr/bin/env python3

# Preprocessing steps follow what is described in the Yin and Wang paper (2014) and is adapted from the code from
# (c) 2010-2011 Nakatni Shuyo / Cybozu Labs Inc. / Cybozu Labs available under the MIT License.


import re
import nltk


stopwords_list = nltk.corpus.stopwords.words('english')
recover_list = {'wa': 'was', 'ha': 'has'}
word_lemmatizer = nltk.WordNetLemmatizer()


def load_corpus(filename):
    """

    :param filename: path to corpus file
    :return: list of lists of tokenized words in documents; each list is a document
    """
    tokenized_corpus = []
    with open(filename, 'r') as f:
        for line in f:
            doc = re.findall(r'\w+(?:\'\w+)?', line)
            # print(doc)
            if len(doc) > 0:
                tokenized_corpus.append(doc)
    return tokenized_corpus


def is_stopword(word):
    """

    :param word: a word token
    :return: True or False
    """
    return word in stopwords_list


def lemmatize(word):
    """

    :param word: a word token
    :return: lemmatized word
    """
    w = word_lemmatizer.lemmatize(word.lower())
    if w in recover_list:
        return recover_list[w]
    return w


def is_too_short(word, threshold):
    return len(word) < threshold


def is_too_long(word, threshold):
    return len(word) > threshold


def load_labels(filename):
    """

    :param filename: file containting true topic labels; 1 label per line
    :return: list of labels
    """
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            # label file has labels from 1 to n, subtract 1 so that labels range from 0 to n - 1 instead for
            # easy indexing
            labels.append(int(line.strip()) - 1)
    return labels


def make_topic_clusters(topic_labels):
    """

    :param topic_labels: list of labels
    :return: list of lists of doc_id's that belong to each topic_label
    """
    num_labels = len(set(topic_labels))
    topic_clusters = [[] for _ in range(num_labels)]

    for doc_id, label_number in enumerate(topic_labels):
        topic_clusters[label_number].append(doc_id)

    return topic_clusters


class Vocabulary:
    """
    Process corpus into int tokens for each tokenized word getting rid of stopwords and words that are too short or
    too long
    """
    def __init__(self, exclude_stopwords=True, exclude_short_long=True, short_word=2, long_word=15):
        self.id_to_word = {}
        self.word_to_id = {}
        self.id_to_doc_frequency = {}  # maps word_id to number of documents the word(id) appears in
        self.exclude_stopwords = exclude_stopwords
        self.exclude_short_long = exclude_short_long
        self.too_short = short_word
        self.too_long = long_word

    def term_to_id(self, word):
        """

        :param word: str word token
        :return: int representation of a word token; None if non-alphabetical or stopword
        """
        term = lemmatize(word)
        if not re.match(r'[a-z]+$', term):
            return None
        if self.exclude_stopwords and is_stopword(term):
            return None
        if self.exclude_short_long:
            if is_too_long(term, self.too_long) or is_too_short(term, self.too_short):
                return None
        if term not in self.word_to_id.keys():
            vocab_id = len(self.id_to_word)
            self.word_to_id[term] = vocab_id
            self.id_to_word[vocab_id] = term
            self.id_to_doc_frequency[vocab_id] = 0
        else:
            vocab_id = self.word_to_id[term]
        return vocab_id

    def doc_to_ids(self, doc):
        """

        :param doc: list of string word tokens
        :return: list of int representations of word tokens, excluding stopwords and non-alphabetical
        """
        doc_as_word_id_list = []
        seen_word_ids = {}
        for term in doc:
            word_id = self.term_to_id(term)
            if word_id is not None:
                doc_as_word_id_list.append(word_id)
                if word_id not in seen_word_ids.keys():
                    seen_word_ids[word_id] = 1
                    self.id_to_doc_frequency[word_id] += 1  # only increment doc count for each unique word id

        return doc_as_word_id_list

    def cut_low_freq(self, indexized_corpus, threshold=1):
        """
        update word_to_id, id_to_word, and id_to_doc_frequency to only contain words that appear at least above the
        threshold

        :param indexized_corpus: list of lists (docs) of integer-represented words
        :param threshold: min number of docs word_id needs to appear in
        :return: updated indexized corpus as list of lists (docs) with updated integer-represented words
        """
        new_id_to_word = {}
        new_id_to_doc_freq = {}
        self.word_to_id = {}
        convert_old_id_to_new_id = {}
        for word_id, term in self.id_to_word.items():
            doc_freq = self.id_to_doc_frequency[word_id]  # num of docs word_id appears in
            if doc_freq > threshold:
                new_word_id = len(new_id_to_word)
                self.word_to_id[term] = new_word_id
                new_id_to_word[new_word_id] = term
                new_id_to_doc_freq[new_word_id] = doc_freq
                convert_old_id_to_new_id[word_id] = new_word_id
        self.id_to_word = new_id_to_word
        self.id_to_doc_frequency = new_id_to_doc_freq

        def convert_word_ids(doc_of_word_ids):
            """

            :param doc_of_word_ids: list of tokenized words represented as int word_ids
            :return: list of tokenized words as int tokens that have been updated
            """
            new_doc = []
            for old_word_id in doc_of_word_ids:
                if old_word_id in convert_old_id_to_new_id:
                    new_doc.append(convert_old_id_to_new_id[old_word_id])
            return new_doc

        return [convert_word_ids(doc) for doc in indexized_corpus]

    def __getitem__(self, word_id):
        """

        :param word_id: int representing word token
        :return: corresponding str word token
        """
        return self.id_to_word[word_id]

    def size(self):
        """

        :return: vocabulary size; number of unique words
        """
        return len(self.id_to_word)

    def is_id_stopword(self, word_id):
        """

        :param word_id: int representing word token
        :return: True/False
        """
        return self.id_to_word[word_id] in stopwords_list


def main():
    # please ignore code below; it's only for testing that this module works as expected
    toy = load_corpus('../data/toy.txt')
    print(toy)

    # process corpora
    vocab = Vocabulary()
    toy_docs = [vocab.doc_to_ids(doc) for doc in toy]
    print(toy_docs)
    print(vocab.id_to_doc_frequency)

    toy_docs = vocab.cut_low_freq(toy_docs)
    print(toy_docs)

    labels = load_labels('../data/label_StackOverflow.txt')
    print(labels)

    true_clusters = make_topic_clusters(labels)
    print(true_clusters[2])


if __name__ == '__main__':
    main()

