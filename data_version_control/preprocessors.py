import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier


class ExtractRelevantColumns:
    """
    Extracts the relevant columns from the dataframe.
    """
    def __init__(self, dataframe, columns):
        self.dataframe = dataframe
        self.columns = columns
    
    def extract(self):
        """
        Extracts the relevant columns from the dataframe.
        """
        return self.dataframe[self.columns]


class CategorizeRelevantData:
    """
    Categorizes the relevant data extracted.
    """
    def __init__(self, dataframe, categories):
        self.dataframe = dataframe
    
    def text_category (p):
        if p > 0:
            return 'positive'
        if p < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def categorize(self):
        """
        Categorizes the relevant data.
        """
        score = pd.Series([text_category(row_value) for row_value in self.dataframe['polarity']])

        Clean_Tweet = pd.concat([Clean_Tweet, score.rename("score")], axis=1)

        return self.dataframe.replace(self.categories)

class VisualizeTheCategorizedData:
    """
    Visualizes the categorized data.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def visualize(self):
        """
        Visualizes the categorized data.
        """

        fig, ax  = plt.subplots(figsize=(20, 10))
        labels = ['neutral', 'positive', 'negative']
        neutral_count = len(self.dataframe[self.dataframe['score'] == "neutral"])
        positive_count = len(self.dataframe[self.dataframe['score'] == "positive"])
        negative_count = len(self.dataframe[self.dataframe['score'] == "negative"])
        sizes = [neutral_count, positive_count, negative_count]
        ax.pie(sizes,startangle=60, labels=labels,autopct='%1.0f%%', colors=["#ededfa","#4caa77","#905850"],textprops={'fontsize': 14})
        ax.add_artist(plt.Circle((0,0),0.3,fc='white'))
        ax.set_title('\npiechart of score') 
        fig.show()

        scoremap = pd.Series([1 if row_value == 'positive' else 0 for row_value in self.dataframe['score']])
        self.dataframe = pd.concat([self.dataframe, scoremap.rename("scoremap")], axis=1)
        self.dataframe['scoremap'] = scoremap
        self.dataframe.reset_index()
        data = self.dataframe[['original_text','score']]

        return data

class SplitDatasets:
    """
    Splits the categorized data into training and testing datasets.
    """
    def __init__(self, dataframe, training_percentage):
        self.dataframe = dataframe
        self.training_percentage = training_percentage
    
    def split(self):
        """
        Splits the categorized data into training and testing datasets.
        """

        train, test = train_test_split(self.dataframe,test_size = self.training_percentage, shuffle=True, stratify=self.dataframe.score,random_state=42)
        train_pos = train[ train['score'] == 'positive']
        train_pos = train_pos['original_text']
        train_neg = train[ train['score'] == 'negative']
        train_neg = train_neg['original_text']
        train_neutral = train[ train['score'] == 'neutral']
        train_neutral = train_neutral['original_text']

        test_pos = test[ test['score'] == 'positive']
        test_pos = test_pos['original_text']
        test_neg = test[ test['score'] == 'negative']
        test_neg = test_neg['original_text']
        test_neutral = test[ test['score'] == 'neutral']
        test_neutral = test_neutral['original_text']

        self.dataframe = self.dataframe.sample(frac=1)
        return train_pos, train_neg, train_neutral, test_pos, test_neg, test_neutral




class ExtractFeatures:
    """
    Extracts features from the categorized data.
    """
    def __init__(self, doc, tweets):
        self.doc = doc
        self.tweets = tweets

    def get_words_in_tweets(tweets):
        all = []
        for (words, score) in tweets:
            all.extend(words)
        return all

    def get_word_features(wordlist):
        wordlist = nltk.FreqDist(wordlist)
        features = wordlist.keys()
        return features


    def extract_features(self):
        document_words = set(self.doc)
        features = {}
        w_features = self.get_word_features(self.get_words_in_tweets(self.tweets))

        for word in w_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
