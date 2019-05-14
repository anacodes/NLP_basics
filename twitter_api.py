from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s 

ckey = 'fA47Fh4oSTqnmNCu9dOfUfvDz'
csecret = 'arT9riow2f9L5B75hDWgJu6L3yUgIYgvPsAHTYPDDS6acOlO2X'
atoken = '826132336253272065-h1yoksZNOBxYsGeN85hDYmzZpg0C22D'
asecret = 'NPmDcW4axL6kTGBFrypVQVHGlTZiceiVgNJvMZtYVEbcG'


class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]

        sentiment_value, confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

            return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["angry"])