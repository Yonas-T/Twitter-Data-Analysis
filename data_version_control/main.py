from preprocessors import SplitDatasets
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier
import pipeline
import preprocessors as pp

tweets = []
stopwords_set = set(stopwords.words("english"))
for index, row in SplitDatasets.split().iterrows():
    words_filtered = [e.lower() for e in row.original_text.split() if len(e) >= 3]
    words_cleaned = [word for word in words_filtered
        if 'http' not in word
        and not word.startswith('@')
        and not word.startswith('#')
        and word != 'RT']
    words_without_stopwords = [word for word in words_cleaned if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.score))
    
    
training_set = nltk.classify.apply_features(pp.ExtractFeatures.extract_features,tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
