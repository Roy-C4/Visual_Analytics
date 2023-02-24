import pandas as pd
import plotly.express as px

def prepare_data():
    # Import data
    df = pd.read_csv('email headers.csv', sep=",")
    print(df.head())
    # Fix columns
    df['From'].astype(str)
    df['Date'] = pd.to_datetime(df['Date'])
    # df['Sender'] = [x.split('@')[0] for x in df['From']]
    # df['Sender'] = [x.replace('.', ', ') for x in df['Sender']]

    
    return df

df = prepare_data()
fig = px.scatter(data_frame=df, x="Date", y="From")