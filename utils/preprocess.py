import pandas as pd

def load_farmer_data(path="data/farmer_advisor_dataset.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df

def load_market_data(path="data/market_researcher_dataset.csv"):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    return df
