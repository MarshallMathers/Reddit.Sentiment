import redditScraper as rs
import pandas as pd
import pickle
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.tokenize.treebank import TreebankWordDetokenizer
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

stops = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


def token_format(strings):

    titles = []
    for line in strings:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stops]
        titles.append(toks)

    refrmt = []
    for line in titles:
        toks2 = TreebankWordDetokenizer().detokenize(line)
        refrmt.append(toks2)

    return refrmt

def train_Bayes():

    ripple = pd.read_table('ripple_train.csv', sep=',')
    btc = pd.read_table('btc_train.csv', sep=',')
    bitcoin = pd.read_table('bitcoin_train.csv', sep=',')
    cryptocurrency = pd.read_table('cryptocurrency_train.csv', sep=',')
    cryptomarkets = pd.read_table('cryptomarkets_train.csv', sep=',')
    ethereum = pd.read_table('ethereum_train.csv', sep=',')
    iota = pd.read_table('iota_train.csv', sep=',')
    litecoin = pd.read_table('litecoin_train.csv', sep=',')
    neo = pd.read_table('neo_train.csv', sep=',')
    stellar = pd.read_table('stellar_train.csv', sep=',')

    headlines = ripple['headline']
    headlines.append(btc['headline'])
    headlines.append(bitcoin['headline'])
    headlines.append(cryptocurrency['headline'])
    headlines.append(cryptomarkets['headline'])
    headlines.append(ethereum['headline'])
    headlines.append(iota['headline'])
    headlines.append(litecoin['headline'])
    headlines.append(neo['headline'])
    headlines.append(stellar['headline'])

    labels = ripple['label']
    labels.append(btc['label'])
    labels.append(bitcoin['label'])
    labels.append(cryptocurrency['label'])
    labels.append(cryptomarkets['label'])
    labels.append(ethereum['label'])
    labels.append(iota['label'])
    labels.append(litecoin['label'])
    labels.append(neo['label'])
    labels.append(stellar['label'])


    reformat = token_format(headlines)

    train = list(zip(reformat, labels))

    dictionary = set(word.lower()
                    for passage in train
                        for word in WordPunctTokenizer().tokenize(passage[0]))



    print("First couple of titles and their associated values:")
    print(train[0])
    print(train[1])
    print(train[2])
    print(train[3])
    
    
    t = [({word: (word in WordPunctTokenizer().tokenize(x[0]))
           for word in dictionary}, x[1]) for x in train]

    classifier = nltk.NaiveBayesClassifier.train(t)

    model = open('bayes_model.pickle', 'wb')
    words = open('dictionary.pickle', 'wb')
    pickle.dump(classifier, model)
    pickle.dump(dictionary, words)
    model.close()
    words.close()


def sentiment_ofTitle(title):

    model = open('bayes_model.pickle', 'rb')
    words = open('dictionary.pickle', 'rb')
    classifier = pickle.load(model)
    dictionary = pickle.load(words)
    model.close()
    words.close()

    test_data = token_format(title)
    print(test_data)
    test_data_features = {word.lower(): (word in WordPunctTokenizer().tokenize(test_data.lower()))
                          for word in dictionary}

    result = classifier.classify(test_data_features)

    if result == "0":
        print("The sentiment of \"{}\" is Neutral".format(title))
    elif result == "1":
        print("The sentiment is \"{}\" is Positive".format(title))
    else:
        print("The sentiment is \"{}\" is Negative".format(title))


def sentiment_ofSubreddit(subreddit, show = False):


    model = open('bayes_model.pickle', 'rb')
    words = open('dictionary.pickle', 'rb')
    classifier = pickle.load(model)
    dictionary = pickle.load(words)
    model.close()
    words.close()
    labels = []
    headlines = rs.get_newSub('{}'.format(subreddit))

    headlines = token_format(headlines)

    for title in headlines:
        test_data = title

        test_data_features = {word.lower(): (word in WordPunctTokenizer().tokenize(test_data.lower()))
                              for word in dictionary}
        labels.append(classifier.classify(test_data_features))


    results = list(zip(headlines, labels))


    sent = pd.Series(labels)

    fig, ax = plt.subplots(figsize=(8, 8))

    counts = sent.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Positive'])
    ax.set_ylabel("Percentage")

    fig.savefig("{}.pdf".format(subreddit))

    if show == True:
        print(labels)

    return counts



