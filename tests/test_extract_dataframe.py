import unittest
import pandas as pd
import sys, os
 
sys.path.append(os.path.abspath(os.path.join('../..')))

from extract_dataframe import read_json
from extract_dataframe import TweetDfExtractor

_, tweet_list = read_json("data/covid19.json")

columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']


class TestTweetDfExtractor(unittest.TestCase):
    """
		A class for unit-testing function in the fix_clean_tweets_dataframe.py file

		Args:
        -----
			unittest.TestCase this allows the new class to inherit
			from the unittest module
	"""

    def setUp(self) -> pd.DataFrame:
        self.df = TweetDfExtractor(tweet_list[:5])
        # tweet_df = self.df.get_tweet_df()         


    def test_find_statuses_count(self):
        self.assertEqual(self.df.find_statuses_count(), [40, 40])

    def test_find_full_text(self):
        text = ['RT @sagt_mit: Merkel schaffte es in 1 Jahr 1 Million "Flüchtlinge" durchzufüttern, jedoch nicht nach 16 Jahren 1 Million Rentner aus der Ar…']


        self.assertEqual(self.df.find_full_text(), text)

    def test_find_sentiments(self):
        self.assertEqual(self.df.find_sentiments(self.df.find_full_text()), ([-0.1, 0.1], [
                         -0.1, -0.4]))

    def test_find_created_time(self):
        created_at = ['Fri Jan 28 21:23:33 +0000 2022', 'Fri Apr 22 14:38:54 +0000 2022']

        self.assertEqual(self.df.find_created_time(), created_at)

    def test_find_source(self):
        source =  ['<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>',
                  '<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>']

        self.assertEqual(self.df.find_source(), source)

    def test_find_screen_name(self):
        name = ['nikitheblogger', 'sagt_mit', 'McMc74078966']
        self.assertEqual(self.df.find_screen_name(), name)

    def test_find_followers_count(self):
        f_count = [3, 3]
        self.assertEqual(self.df.find_followers_count(), f_count)

    def test_find_friends_count(self):
        friends_count = [12, 193]
        self.assertEqual(self.df.find_friends_count(), friends_count)

    def test_find_is_sensitive(self):
        self.assertEqual(self.df.is_sensitive(), [None, None])

    def test_find_favourite_count(self):
        self.assertEqual(self.df.find_favourite_count(), [1985, 0])

    def test_find_retweet_count(self):
        self.assertEqual(self.df.find_retweet_count(), [505, 4])

 
    def test_find_location(self):
        self.assertEqual(self.df.find_location(), ['Gnaden Vürttemberg ', 'Deutschland'])

if __name__ == '__main__':
	unittest.main()

    