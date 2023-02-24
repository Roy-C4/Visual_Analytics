import pandas as pd

def prepare_data():
    # Import data
    df = pd.read_csv('/email headers.csv', sep=",")

    # Fix columns
    df['Date'] = pd.to_datetime(df['Date'])
    df['Sender'] = [x.split('@')[0] for x in df['From']]
    df['Sender'] = [x.replace('.', ', ') for x in df['Sender']]

    df_new = df[['Sender', 'Date']].copy()

    return df_new