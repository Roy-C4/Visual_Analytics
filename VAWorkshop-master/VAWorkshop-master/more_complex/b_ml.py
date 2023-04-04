from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor, LinearRegression
import pandas as pd
from sklearn.metrics import mean_squared_error

def do_ml(
        df,
        remove=['Country name', 'Standard error of ladder score'],
        target=['Ladder score']
):
    keep_cols = [c for c in df.columns if c not in remove + target]
    df3 = df[keep_cols]

    df4 = pd.get_dummies(df3, prefix_sep='::')
    # for x in categorical:
    # le_encoders[x] = preprocessing.LabelEncoder()
    # df3[x] = le_encoders[x].fit_transform(df3[x])

    keep_cols = [c for c in df4.columns if c not in target]
    X, y = df4[keep_cols], df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    gnb = LinearRegression()
    # gnb = LinearRegression(random_state=1)
    gnb.fit(X_train, y_train.values.ravel())
    print(mean_squared_error(y_test, gnb.predict(X_test)))

    return gnb, X_train, X_test, y_train, y_test, X, y
