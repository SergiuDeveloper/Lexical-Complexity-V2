#!/usr/bin/python3

TRAIN_DATASET_PATH = './../CompLex/train/lcp_single_train.tsv'
TEST_DATASET_PATH = './../CompLex/test-labels/lcp_single_test.tsv'

from nltk.corpus import wordnet
from nltk.wsd import lesk
import pandas as pd
import math


def load_dataframe(dataframe_path):
    return pd.read_csv(dataframe_path, sep='\t')

def get_tokens_POS(sentence_token_complexity_pairs):
    tokens_POS = [(sentence, lesk(sentence, str(token)), complexity) for sentence, token, complexity in sentence_token_complexity_pairs]
    #for sentence, token, complexity
    return [(sentence, token, complexity) for sentence, token, complexity in tokens_POS if token is not None]


train_df = load_dataframe(TRAIN_DATASET_PATH)
test_df = load_dataframe(TEST_DATASET_PATH)

train_tokens_POS = get_tokens_POS(list(train_df[['sentence', 'token', 'complexity']].itertuples(index=False)))
test_tokens_POS = get_tokens_POS(list(test_df[['sentence', 'token', 'complexity']].itertuples(index=False)))

for _, test_token, test_complexity in test_tokens_POS:
    path_similarities = []
    train_complexities = []
    for _, train_token, train_complexity in train_tokens_POS:
        path_similarity = wordnet.path_similarity(test_token, train_token)
        if path_similarity is None:
            path_similarity = 0
        path_similarities.append(math.fabs(path_similarity))
        train_complexities.append(train_complexity)
    
    total_path_similarities = sum(path_similarities)
    predicted_complexity = 0
    for i in range(len(path_similarities)):
        term_relevance = path_similarities[i] / len(path_similarities)
        predicted_complexity += train_complexities[i] * term_relevance

    print('Word: \"{}\"'.format(test_token.lemmas()[0].name()))
    print('Complexity:', test_complexity)
    print('Predicted complexity:', predicted_complexity)
    print()