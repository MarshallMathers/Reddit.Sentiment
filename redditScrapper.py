import praw
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')

stops = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

console_width = 520
pd.set_option('display.width', console_width)
np.set_printoptions(linewidth=console_width)
pd.set_option('display.max_columns', 100)

reddit = praw.Reddit(client_id='68JOg907T0NRZg',
                     client_secret='itgB8kUjsN-_EWDov6vA8ildMww',
                     user_agent='Sentiment Analysis')


def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stops]
        tokens.extend(toks)

    return tokens


def get_newTitle(subreddit):

    subreddit = reddit.subreddit('{}'.format(subreddit))
    new = subreddit.new(limit=1)

    for submission in new:
        return submission.title

def get_sentimentSubreddit(subreddit):

    subreddit = reddit.subreddit('{}'.format(subreddit))
    hot_btc = subreddit.hot(limit=None)
    headlines = set()

    for submission in hot_btc:
        headlines.add(submission.title)

    sia = SIA()
    results = []

    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    df = pd.DataFrame.from_records(results)

    df['label'] = 0
    df.loc[df['compound'] > 0.2, 'label'] = 1
    df.loc[df['compound'] < -0.2, 'label'] = -1

    # print(df.head(5))
    return df

def sample_Headlines(dataframe):
    df = dataframe
    print("Positive headlines:\n")
    pos = df[df['label'] == 1].headline
    print(pos)

    print("\nNegative headlines:\n")
    print(list(df[df['label'] == -1].headline)[:5])

def sentiment_Stats(dataframe):
    df = dataframe
    print('\n')
    print("We can see that looking at the past 100 titles or so that:")
    for val, cnt in df.label.value_counts().iteritems():
        print("There are", cnt, "titles that have a sentiment of", val)
    print('\n')
    print("If we wanted to look at the distribution, it shows that: ")
    for val, cnt in df.label.value_counts(normalize=True).iteritems():
        print(int(cnt*100), "% are titles that have a sentiment of", val)


def top20_Positivewords(dataframe):
    df = dataframe
    pos_lines = list(df[df.label == 1].headline)

    pos_tokens = process_text(pos_lines)
    pos_freq = nltk.FreqDist(pos_tokens)

    print(pos_freq.most_common(20))

def top20_Negativewords(dataframe):
    df = dataframe
    pos_lines = list(df[df.label == -1].headline)

    pos_tokens = process_text(pos_lines)
    pos_freq = nltk.FreqDist(pos_tokens)

    print(pos_freq.most_common(20))

def plot_WordDistribution(dataframe):
    df = dataframe
    pos_lines = list(df[df.label == 1].headline)

    pos_tokens = process_text(pos_lines)
    pos_freq = nltk.FreqDist(pos_tokens)

    y_val = [x[1] for x in pos_freq.most_common()]

    fig = plt.figure(figsize=(10, 5))
    plt.plot(y_val)

    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Word Frequency Distribution (Positive)")
    plt.show()

def plot_Total_Distribution(data):

    df = data
    fig, ax = plt.subplots(figsize=(8, 8))

    counts = df.label.value_counts(normalize=True) * 100

    sns.barplot(x=counts.index, y=counts, ax=ax)

    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_ylabel("Percentage")

    plt.show()
