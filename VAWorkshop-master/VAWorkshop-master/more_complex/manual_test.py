# %%
import pandas as pd

df = pd.read_csv('./world-happiness-report-2021.csv')

#%% statistical investigation
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Profiling Report")
profile.to_file('file.html')
#%% cleaning
df2 = df[~df['Time'].isna()]
df2 = df2[~df2['Road_Surface_Conditions'].isna()]
df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y')
df2 = df2[df2['Number_of_Casualties'] < 9]
df2 = df2[df2['Number_of_Vehicles'] < 9]

#%%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

categorical_columns = [
    'Road_Type',
    'Light_Conditions',
    'Weather_Conditions',
    'Road_Surface_Conditions'
]
df3 = df2[:]
le_encoders = {}
for x in categorical_columns:
    le_encoders[x] = preprocessing.LabelEncoder()
    df3[x] = le_encoders[x].fit_transform(df3[x])


X, y = df3.iloc[:, :-1], df3.iloc[:, -1]
X = X.drop(['Longitude', 'Latitude', 'Date'], axis=1)
X['Time'] = X['Time'].str.replace(':', '')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%%
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
gnb.score(X_train, y_train)
