import praw
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


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


def get_newSub(subreddit):

    subreddit = reddit.subreddit('{}'.format(subreddit))
    new = subreddit.new(limit=750)
    headlines = []

    for submission in new:
        headlines.append(submission.title)

    return headlines


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

