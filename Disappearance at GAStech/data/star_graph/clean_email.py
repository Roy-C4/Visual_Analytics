import pandas as pd
import csv

#creates new email file with senders and receivers having a one-to-one relationship

df = pd.read_csv('D:\roy\selected\TU\books\Q3\visual_analytics\data-1\data\graph\email headers.csv')
df = df.head(100)

df = df.assign(Receivers=df['To'].str.split(',')).explode('Receivers')

print(df['To'])

df = df.drop( 'To', axis =1)

df.to_csv('new_email.csv', index = False)


