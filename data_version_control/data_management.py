import pandas as pd
import config
def load_data(file_name):
    """
    Loads data from a csv file.
    """
    return pd.read_csv(config.DATAPATH + file_name)
