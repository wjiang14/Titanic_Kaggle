import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import cross_validation

# preprocessing
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data_train['Sex'] = data_train['Sex'].map({'male': 1, 'female': 0})
data_test['Sex'] = data_test['Sex'].map({'male': 1, 'female': 0})
# Pclass_dummy1 = pd.get_dummies(data_train['Pclass'])
# Pclass_dummy2 = pd.get_dummies(data_test['Pclass'])
# Pclass_dummy1.columns = ['Class1', 'Class2', 'Class3']
# Pclass_dummy2.columns = ['Class1', 'Class2', 'Class3']
# data_train.drop('Pclass', axis=1, inplace=True)
# data_test.drop('Pclass', axis=1, inplace=True)
# data_train = data_train.join(Pclass_dummy1)
# data_test = data_test.join(Pclass_dummy2)

# fill Nan data at Fare item in test data
fareMean = data_test['Fare'].mean()
fareStd = data_test['Fare'].std()
# data_test.loc[data_test['Fare'].isnull(), 'Fare'] = np.random.randint(0 if fareMean - fareStd < 0 else fareMean - fareStd, fareMean + fareStd)
data_test.loc[data_test['Fare'].isnull(), 'Fare'] = fareMean


# there're 714 non-null value in data age item
# use random forest regressor to estimate age items
def set_missing_age(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # set age item as know and unknow
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y: target
    y = known_age[:, 0]
    # X: feature
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1:])
    df.loc[df['Age'].isnull(), 'Age'] = predictedAges
    return df, rfr

data_train, rfr = set_missing_age(data_train)
data_test, rfr = set_missing_age(data_test)

# rescale Fare item for train data
sc = StandardScaler()
ageScaler = sc.fit(data_train['Age'])
fareScaler = sc.fit(data_train['Fare'])
data_train['Age'] = sc.fit_transform(data_train['Age'], ageScaler)
data_train['Fare'] = sc.fit_transform(data_train['Fare'], fareScaler)


# recale Fare item for test data
sc = StandardScaler()
ageScaler_test = sc.fit(data_test['Age'])
fareScaler_test = sc.fit(data_test['Fare'])
data_test['Age'] = sc.fit_transform(data_test['Age'], ageScaler_test)
data_test['Fare'] = sc.fit_transform(data_test['Fare'], fareScaler_test)

# test test data using Logistic regression

X_train, X_test, y_train, y_test = train_test_split(data_train.iloc[:, 1:].values, data_train['Survived'].values, test_size=0.3, random_state=0)
lr = LogisticRegression(C=100.0, random_state=1)
lr.fit(X_train, y_train)
score = lr.score(X_test, y_test)
print score
print lr.predict(X_test)

# cross validation
# print cross_validation.cross_val_score(lr, X_train, y_train, cv=5)

ToTest = data_test.iloc[:, 1:].values
prediction = lr.predict(ToTest)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': prediction.astype(np.int32)})
result.to_csv("/Users/wei/Documents/data_sci/kaggle/titanic2/LR_predictions_random_forest.csv", index=False)