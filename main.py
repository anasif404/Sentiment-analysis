# IMPORTING NEEDED LIBS
import string
import nltk
import re

import numpy
import pandas as pd
import gensim
import numpy as np
import tweepy
import configparser

from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tabulate import tabulate
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# INITIALIZE VARIABLES FOR PREPROCESSING
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("english")
stop = stopwords.words('english')

# CONFIG
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# AUTHENTICATION
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# SEARCH TWEETS

keywords = 'gaza'
limit = 200
tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)

columns = ['User', 'Tweet', 'Sentiment']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text, ''])

df = pd.DataFrame(data, columns=columns)

# READING DATASET AND MAKING A SMALLER COPY CALLED DF(DATAFRAME)
# NAMING THE HEADERS
ds = pd.read_csv('twitter_validation.csv', names=['Tweet ID', 'Entity', 'Sentiment', 'Tweet'])
sdf = ds.copy()
sdf = sdf.iloc[:1000]
sdf["Tweet"] = sdf["Tweet"].astype('str')
print(sdf.dtypes)


# REMOVING PUNCTUATION FUNCTION
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


# REMOVING SPECIAL CHARACTER FUNCTION
def remove_characters(text):
    text = text.strip()
    PATTERN = '[^a-zA-Z ]'
    filtered_text = re.sub(PATTERN, '', text)
    return filtered_text


# REMOVING REPEATED CHARACTERS FUNCTION
def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

sdf["Tweet"] = sdf["Tweet"].astype(str).str.lower()  # LOWER CASING ALL TWEETS
sdf["Tweet"] = sdf["Tweet"].apply(remove_punctuations)  # REMOVING PUNCTUATIONS
sdf["Tweet"] = sdf["Tweet"].apply(remove_characters)  # REMOING SPECIAL CHARACTERS
sdf["Tweet"] = sdf["Tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))  # REMOVING STOP WORDS
sdf["Tweet"] = sdf["Tweet"].apply(word_tokenize)  # TOKENIZING EACH WORD IN THE TWEETS
sdf["Tweet"] = sdf['Tweet'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])  # LEMMETIZING ALL TOKENIZED WORDS
sdf["Tweet"] = sdf["Tweet"].apply(remove_repeated_characters)  # REMOVING REPEATED WORDS
sdf["Tweet"] = sdf['Tweet'].apply(lambda x: [stemmer.stem(y) for y in x])  # STEMMING WORDS

df["Tweet"] = df["Tweet"].astype(str).str.lower()  # LOWER CASING ALL TWEETS
df["Tweet"] = df["Tweet"].apply(remove_punctuations)  # REMOVING PUNCTUATIONS
df["Tweet"] = df["Tweet"].apply(remove_characters)  # REMOING SPECIAL CHARACTERS
df["Tweet"] = df["Tweet"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))  # REMOVING STOP WORDS
df["Tweet"] = df["Tweet"].apply(word_tokenize)  # TOKENIZING EACH WORD IN THE TWEETS
df["Tweet"] = df['Tweet'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])  # LEMMETIZING ALL TOKENIZED WORDS
df["Tweet"] = df["Tweet"].apply(remove_repeated_characters)  # REMOVING REPEATED WORDS
df["Tweet"] = df['Tweet'].apply(lambda x: [stemmer.stem(y) for y in x])  # STEMMING WORDS

# PRINTING TABLE LAYOUT FOR DATASET
print(tabulate(df, headers='keys', tablefmt='psql'))


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range, max_features=5000)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range,
                                 max_features=5000)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv.__getitem__(word))
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index_to_key)
    features = [average_word_vectors(sentence, model, vocabulary, num_features) for sentence in corpus]
    return np.array(features)


def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)]
                   if tfidf_vocabulary.get(word)
                   else 0 for word in words]
    word_tfidf_map = {word: tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}

    feature_vector = np.zeros((num_features,), dtype="float64")
    vocabulary = set(model.wv.index_to_key)
    wts = 0.
    for word in words:
        if word in vocabulary:
            word_vector = model.wv.__getitem__(word)
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)

    return feature_vector


def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors,
                                            tfidf_vocabulary, model, num_features):
    docs_tfidfs = [(doc, doc_tfidf)
                   for doc, doc_tfidf
                   in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                           model, num_features)
                for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)


# plt.rcParams["figure.figsize"] = [8, 10]
# df.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
# plt.show()

'''class1 = df[df["Sentiment"] == 1]
class2 = df[df["Sentiment"] == 2]
class3 = df[df["Sentiment"] == 3]
class4 = df[df["Sentiment"] == 4]

downsample_class1 = resample(class1,
             replace=True,
             n_samples=len(class2),
             random_state=42)

downsample_class3 = resample(class3,
             replace=True,
             n_samples=len(class2),
             random_state=42)

downsample_class4 = resample(class4,
             replace=True,
             n_samples=len(class2),
             random_state=42)

dfs = pd.concat([class2, downsample_class1,downsample_class3,downsample_class4])
dfs.groupby('Sentiment').size().plot(kind='pie',
                                       y = "Sentiment",
                                       label = "Sentiment",
                                       autopct='%1.1f%%')'''

#X = sdf.Tweet
#Y = sdf.Sentiment
#train_X, train_Y, test_X, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
train_X = sdf.Tweet
train_Y = sdf.Sentiment
test_X = df.Tweet
test_Y = df.Sentiment
print(train_X.shape)
print(test_X.shape)
print(train_Y.shape)
print(test_Y.shape)

train_corpus = train_X.astype("U")
test_corpus = test_X.astype('U')
train_labels, test_labels = train_Y, test_Y


def display_features(features, feature_names):
    df = pd.DataFrame(data=features,
                      columns=feature_names)
    print("FEATURES:\n", df)


feature_set = []

# BAG OF WORDS
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
feature_set.append(('Bag of words features', bow_train_features, bow_test_features))
feature_names = bow_vectorizer.get_feature_names_out()
print("BOW:\n", feature_names)

# TF-IDF
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(test_corpus)
feature_set.append(('Tfidf features', tfidf_train_features, tfidf_test_features))
feature_names = tfidf_vectorizer.get_feature_names_out()
print("TF-IDF:\n", feature_names)

features = tfidf_train_features.todense()
features_tfidf = np.round(features, 2)
display_features(features_tfidf[0:5], feature_names)

tokenized_train = [nltk.word_tokenize(text) for text in train_corpus]
tokenized_test = [nltk.word_tokenize(text) for text in test_corpus]

# WORD2VEC MODEL
model = gensim.models.Word2Vec(tokenized_train, vector_size=100, window=100, min_count=2, sample=1e-3)
# AVERAGED WORD VECTOR
avg_wv_train_features = averaged_word_vectorizer(corpus=tokenized_train, model=model, num_features=100)
avg_wv_test_features = averaged_word_vectorizer(corpus=tokenized_test, model=model, num_features=100)
feature_set.append(('Averaged word vector features', avg_wv_train_features, avg_wv_test_features))

print("AWV:\n", np.round(avg_wv_test_features, 3))

# TF-IDF WEIGHTED AVERAGE
vocab = tfidf_vectorizer.vocabulary_
tfidf_wv_train_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_train,
                                                                  tfidf_vectors=tfidf_train_features,
                                                                  tfidf_vocabulary=vocab, model=model, num_features=100)
tfidf_wv_test_features = tfidf_weighted_averaged_word_vectorizer(corpus=tokenized_test,
                                                                 tfidf_vectors=tfidf_test_features,
                                                                 tfidf_vocabulary=vocab, model=model, num_features=100)
feature_set.append(('Tfidf weighted averaged word vector features', tfidf_wv_train_features, tfidf_wv_test_features))

print("TF-IDF:\n", np.round(tfidf_wv_test_features, 3))


def get_metrics(true_labels, predicted_labels):
    print('Accuracy: ', accuracy_score(true_labels, predicted_labels))
    print(classification_report(true_labels, predicted_labels))


def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)
    train_predictions = classifier.predict(train_features)
    test_predictions = classifier.predict(test_features)
    print('Training set performance:')
    get_metrics(true_labels=train_labels, predicted_labels=train_predictions)
    print('Test set performance:')
    get_metrics(true_labels=test_labels, predicted_labels=test_predictions)
    return test_predictions


models = []
models.append(('MNB', MultinomialNB()))
models.append(('SVC', SVC()))
models.append(('SGD', SGDClassifier()))
models.append(('RFC', RandomForestClassifier()))

predict_result = []
'''for m_name, model in models:
    print('==================================================')
    print('Model:', m_name)
    for f_name, train_features, test_features in feature_set:
        if m_name == 'MNB':
            if (f_name == 'Averaged word vector features' or f_name == 'Tfidf weighted averaged word vector features'):
                break
        print(f_name)
        predictions = train_predict_evaluate_model(model, train_features, train_labels, test_features, test_labels)
        print('--------------------------------------------------')
        predict_result.append((m_name, f_name, predictions))'''

print()
# CONFUSION MATRIX
'''for m_name, f_name, predictions in predict_result:
    print("Model {} with {}:".format(m_name, f_name))
    cm = confusion_matrix(test_labels, predictions)
    print(pd.DataFrame(cm, index=range(1, 5), columns=range(1, 5)))
    #sn.set(font_scale=1.4)  # for label size
    #sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    #plt.show()'''


def get_optimal_parameters(classifier, param_grid,
                           train_features, train_labels,
                           test_features, test_labels):
    classifier_cv = GridSearchCV(classifier, param_grid, cv=5)
    classifier_cv.fit(train_features, train_labels)
    test_predictions = classifier_cv.predict(test_features)
    print("Tuned Parameter: {}".format(classifier_cv.best_params_))
    print("Tuned Score: {}".format(classifier_cv.best_score_))
    print()
    print('Test set performance:')
    get_metrics(true_labels=test_labels, predicted_labels=test_predictions)
    # PRINT OPTIMAL PARAMETERS
    return classifier_cv


# MNB
'''alpha_space = np.logspace(-3, 3, 10)
param_grid = {'alpha': alpha_space}
print('MNB')
MNB_bow_model = get_optimal_parameters(classifier=MultinomialNB(),
                                    param_grid=param_grid,
                                    train_features=bow_train_features,
                                    train_labels=train_labels,
                                    test_features=bow_test_features,
                                    test_labels=test_labels)'''
# SVC
c_space = np.logspace(-3, 2, 10)
param_grid = {'C': c_space, 'kernel': ('linear', 'rbf')}
print('SVC')
LSVC_tfidf_model = train_predict_evaluate_model(classifier=SVC(),
                                                train_features=tfidf_train_features,
                                                train_labels=train_labels,
                                                test_features=tfidf_test_features,
                                                test_labels=test_labels)
df["Sentiment"] = LSVC_tfidf_model
print(tabulate(df, headers='keys', tablefmt='psql'))
plt.rcParams["figure.figsize"] = [8, 10]
df.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.show()
# SGD
'''l1_space = np.linspace(0, 1, 20)
param_grid = {'l1_ratio': l1_space}
print('SGD')
SGD_tfidf_model = get_optimal_parameters(classifier=SGDClassifier(),
                                         param_grid=param_grid,
                                         train_features=tfidf_train_features,
                                         train_labels=train_labels,
                                         test_features=tfidf_test_features,
                                         test_labels=test_labels)'''
# RANDOM FOREST
'''n_options = [10, 20, 50, 100, 200]
sample_leaf_options = [1, 5, 10, 50, 100, 200, 500]
param_grid = {'n_estimators': n_options, 'min_samples_leaf': sample_leaf_options}
print('RANDOM FOREST')
RFC_avgwv_model = get_optimal_parameters(classifier=RandomForestClassifier(),
                                         param_grid=param_grid,
                                         train_features=avg_wv_train_features,
                                         train_labels=train_labels,
                                         test_features=avg_wv_test_features,
                                         test_labels=test_labels)'''
