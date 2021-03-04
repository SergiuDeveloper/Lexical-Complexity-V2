#!/usr/bin/python3

TRAIN_DATASET_PATH = './../CompLex/train/lcp_single_train.tsv'
TEST_DATASET_PATH = './../CompLex/test-labels/lcp_single_test.tsv'
FASTTEXT_MODEL_PATH = './models/FastText/cbow'
LOGS_FOLDER_PATH = './logs'
LANGUAGE = 'english'

from gensim.models.fasttext import FastText
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import itertools
import re
from datetime import datetime


def load_dataframe(dataframe_path):
    return pd.read_csv(dataframe_path, sep='\t')

def preprocess_data(sentences, language):
    language_stopwords = set(stopwords.words(language))

    sentences = [re.sub(r'[^a-zA-Z\s]', '', sentence).lower() for sentence in sentences]
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [
        [word for word in sentence if word not in language_stopwords]
        for sentence in tokenized_sentences
    ]

    return tokenized_sentences

def create_FastText_model(model_path, skip_gram, epochs_count, tokenized_sentences):
    try:
        model = FastText.load(model_path)
    except:
        model = FastText(min_n=1, hs=True, alpha=0.1, min_alpha=0.1, sg=skip_gram)
        model.build_vocab(sentences=tokenized_sentences)
        model.train(sentences=tokenized_sentences, total_examples=len(tokenized_sentences), epochs=epochs_count)

        model.save(model_path)

    return model

def compute_mse(y_true, y_pred):
    return sum([((y_true[i] - y_pred[i]) ** 2) for i in range(len(y_true))]) / len(y_true)


train_df = load_dataframe(TRAIN_DATASET_PATH)
test_df = load_dataframe(TEST_DATASET_PATH)

wordnet_synset_examples = [synset.examples() for synset in wordnet.all_synsets()]
wordnet_synset_examples = list(itertools.chain.from_iterable(wordnet_synset_examples))
wordnet_synset_examples_tokenized = preprocess_data(wordnet_synset_examples, LANGUAGE)

fasttext_model = create_FastText_model(FASTTEXT_MODEL_PATH, False, 1, wordnet_synset_examples_tokenized)






date_hour = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
with open('{}/{}.txt'.format(LOGS_FOLDER_PATH, date_hour), 'w') as logs_file:
    nn_train_features = []
    nn_train_labels = []
    computed_complexities = []

    iter = 0
    for test_token, test_complexity in test_df[['token', 'complexity']].itertuples(index=False):
        if type(test_token) is not str:
            continue
        test_token = test_token.lower()
        complexity = 0
        similarities_sum = 0
        for train_token, train_complexity in train_df[['token', 'complexity']].itertuples(index=False):
            if type(train_token) is not str:
                continue
            train_token = train_token.lower()
            similarity = fasttext_model.wv.similarity(test_token, train_token)
            complexity += similarity * train_complexity
            similarities_sum += similarity
        complexity /= similarities_sum
        computed_complexities.append(complexity)
        nn_train_features.append(complexity)
        nn_train_labels.append(test_complexity)

        iter += 1
        log = '{} / {} ~ "{}" ~ MSE={}'.format(iter, len(test_df), test_token, compute_mse(computed_complexities, nn_train_labels))
        print(log)
        logs_file.write('{}\n'.format(log))