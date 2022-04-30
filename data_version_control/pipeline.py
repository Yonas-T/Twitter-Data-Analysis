from sklearn.pipeline import Pipeline
import preprocessors as pp
import data_management as dm



data_loaded = dm.load_data('Economic_Twitter_Data.json')



price_pipe = Pipeline(
    [
        ('Extract Relevant Column',pp.ExtractRelevantColumns(dataframe= data_loaded, columns=['original_text','polarity', 'subjectivity'])),
        ('Categorize Relevant Data', pp.CategorizeRelevantData()),
        ('Visualize Categorized Data', pp.VisualizeTheCategorizedData()),
        ('Split Datasets', pp.SplitDatasets()),
        ('Extract Features', pp.ExtractFeatures()),
        ]
)
