import redditScraper as rs
import analyzer as an
import seaborn as sns
sns.set(style='darkgrid', context='talk', palette='Dark2')


def write_new_Bayes():
    an.train_Bayes()

# Call this function if you want to train Bayes again. (Not Recommended!)
# write_new_Bayes()

def sample_bayes(subreddit):
    sampleTitle = rs.get_newTitle('{}'.format(subreddit))
    an.sentiment_ofTitle(sampleTitle)


def get_Latest_sentiment():
    print("The current spread of Btc is:")
    print(an.sentiment_ofSubreddit('btc'))
    print(50 * '-')
    print("The current spread of Litecoin is:")
    print(an.sentiment_ofSubreddit('litecoin'))
    print(50 * '-')
    print("The current spread of Stellar is:")
    print(an.sentiment_ofSubreddit('stellar'))
    print(50 * '-')
    print("The current spread of Ripple is:")
    print(an.sentiment_ofSubreddit('ripple'))
    print(50 * '-')
    print("The current spread of Ethereum is:")
    print(an.sentiment_ofSubreddit('ethereum'))

get_Latest_sentiment()
