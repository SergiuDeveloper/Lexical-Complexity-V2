#!/usr/bin/python3

TRAIN_DATASET_PATH = './../CompLex/train/lcp_single_train.tsv'
TEST_DATASET_PATH = './../CompLex/test-labels/lcp_single_test.tsv'
FASTTEXT_CBOW_MODEL_PATH = './models/cbow'
FASTTEXT_SG_MODEL_PATH = './models/sg'
LOGS_FOLDER_PATH = './logs'
LANGUAGE = 'english'

from gensim.models.fasttext import FastText
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.wsd import lesk
import pandas as pd
import itertools
import re
from datetime import datetime


def load_dataframe(dataframe_path):
    return pd.read_csv(dataframe_path, sep='\t')

def get_tokens_POS(sentence_token_complexity_pairs):
    stemmer = SnowballStemmer(LANGUAGE)
    tokens_POS = [(sentence, lesk(sentence, str(token)), complexity) for sentence, token, complexity in sentence_token_complexity_pairs]

    for i in range(len(tokens_POS)):
        sentence, token, complexity = tokens_POS[i]
        if token is None:
            word = sentence_token_complexity_pairs[i][1]
            token = lesk(sentence, stemmer.stem(word))

            if token is None:
                tokens_POS[i] = (sentence, word, complexity)
            else:
                tokens_POS[i] = (sentence, token, complexity)

    return [(sentence, token, complexity) for sentence, token, complexity in tokens_POS]

def preprocess_data(sentences, language):
    language_stopwords = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()

    sentences = [re.sub(r'[^a-zA-Z\s]', '', sentence).lower() for sentence in sentences]
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    tokenized_sentences = [
        [lemmatizer.lemmatize(word) for word in sentence if word not in language_stopwords]
        for sentence in tokenized_sentences
    ]

    return tokenized_sentences

def create_FastText_model(skip_gram, tokenized_sentences, model_path):
    try:
        model = FastText.load(model_path)
    except:
        model = FastText(min_count=1, window=5, sg=skip_gram)
        model.build_vocab(sentences=tokenized_sentences)
        model.train(sentences=tokenized_sentences, total_examples=len(tokenized_sentences), vector_size=5, epochs=10)

        model.save(model_path)

    return model


train_df = load_dataframe(TRAIN_DATASET_PATH)
test_df = load_dataframe(TEST_DATASET_PATH)

wordnet_synset_examples = [synset.examples() for synset in wordnet.all_synsets()]
wordnet_synset_examples = list(itertools.chain.from_iterable(wordnet_synset_examples))
wordnet_synset_examples_tokenized = preprocess_data(wordnet_synset_examples, LANGUAGE)

fasttext_model = create_FastText_model(False, wordnet_synset_examples_tokenized, FASTTEXT_CBOW_MODEL_PATH)

train_tokens_POS = get_tokens_POS(list(train_df[['sentence', 'token', 'complexity']].itertuples(index=False)))
test_tokens_POS = get_tokens_POS(list(test_df[['sentence', 'token', 'complexity']].itertuples(index=False)))



date_hour = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
logs_file = open('{}/{}.txt'.format(LOGS_FOLDER_PATH, date_hour), 'w')

complexities_squared_errors = []
for _, test_token, test_complexity in test_tokens_POS:
    similarities = []
    train_complexities = []
    for _, train_token, train_complexity in train_tokens_POS:
        if isinstance(train_token, str):
            continue

        if isinstance(test_token, str):
            similarity = fasttext_model.wv.similarity(test_token, train_token.lemmas()[0].name())
        else:
            similarity = wordnet.path_similarity(test_token, train_token)

        if similarity is None:
            similarity = 0
        similarities.append(similarity)
        train_complexities.append(train_complexity)
    
    predicted_complexity = 0
    for i in range(len(similarities)):
        term_relevance = similarities[i] / len(similarities) * 10
        predicted_complexity += train_complexities[i] * term_relevance

    complexities_squared_errors.append((predicted_complexity - test_complexity) ** 2)

    print('Word: \"{}\"'.format(test_token.lemmas()[0].name() if not isinstance(test_token, str) else test_token))
    print('Complexity:', test_complexity)
    print('Predicted complexity:', predicted_complexity)
    print('MSE:', sum(complexities_squared_errors) / len(complexities_squared_errors))
    print()

    logs_file.write('Word: \"{}\"\n'.format(test_token.lemmas()[0].name() if not isinstance(test_token, str) else test_token))
    logs_file.write('Complexity: {}\n'.format(test_complexity))
    logs_file.write('Predicted complexity: {}\n'.format(predicted_complexity))
    logs_file.write('MSE: {}\n'.format(sum(complexities_squared_errors) / len(complexities_squared_errors)))
    logs_file.write('\n')

logs_file.close()