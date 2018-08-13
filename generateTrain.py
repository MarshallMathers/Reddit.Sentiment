import praw
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

reddit = praw.Reddit(client_id='68JOg907T0NRZg',
                     client_secret='itgB8kUjsN-_EWDov6vA8ildMww',
                     user_agent='Sentiment Analysis')


def generate_trainData(subreddit):

    subreddit = reddit.subreddit('{}'.format(subreddit))
    top_year = subreddit.top('year', limit=100)
    headlines = set()

    for submission in top_year:
        headlines.add(submission.title)

    # for submission in hot_eth:
    #    headlines.add(submission.title)

    sia = SIA()
    results = []

    for line in headlines:
        pol_score = sia.polarity_scores(line)
        pol_score['headline'] = line
        results.append(pol_score)

    df = pd.DataFrame.from_records(results)

    df['label'] = '0'
    df.loc[df['compound'] > 0.1, 'label'] = '1'
    df.loc[df['compound'] < -0.1, 'label'] = '-1'

    df2 = df[['headline', 'label']]
    filename = ('{}_train.csv'.format(subreddit))
    df2.to_csv(filename, mode='w', encoding='utf-8', index=False)

# Generates 10 test files that are used for training Niave Bayes

"""
generate_trainData("ripple")
generate_trainData("iota")
generate_trainData("ethereum")
generate_trainData("btc")
generate_trainData("litecoin")


generate_trainData("cryptocurrency")
generate_trainData("cryptomarkets")
generate_trainData("stellar")
generate_trainData("neo")
generate_trainData("bitcoin")
"""
