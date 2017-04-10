import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
data_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
data_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# deal with Sex, male:1; female:0
data_train['Sex'] = data_train['Sex'].map({'male': 1, 'female': 0}).astype(int)
data_test['Sex'] = data_test['Sex'].map({'male': 1, 'female': 0}).astype(int)

# deal with Pclass
PclassDummiesTrain = pd.get_dummies(data_train['Pclass'])
PclassDummiesTest = pd.get_dummies(data_train['Pclass'])
PclassDummiesTrain.columns = ['Class1', 'Class2', 'Class3']
PclassDummiesTest.columns = ['Class1', 'Class2', 'Class3']
data_train.drop('Pclass', axis=1, inplace=True)
data_test.drop('Pclass', axis=1, inplace=True)
data_train = data_train.join(PclassDummiesTrain)
data_test = data_test.join(PclassDummiesTest)

# age column has Nan, try to use, 714 notnull out of 891 data at Age columns
# make a easy judge: if age less than 15, it's 1. otherwise set it to 0, i
# f it does not work well, consider the distribution
# set Nan value in Age to a random number

averageAgeTrain = data_train['Age'].mean()
stdAgeTrain = data_train['Age'].std()
N_NanAgeTrain = data_train['Age'].isnull().sum()

averageAgeTest = data_test['Age'].mean()
stdAgeTest = data_test['Age'].std()
N_NanAgeTest = data_test['Age'].isnull().sum()

rand_train = np.random.randint(0 if averageAgeTrain - stdAgeTrain < 0 else averageAgeTrain - stdAgeTrain, averageAgeTrain + stdAgeTrain, N_NanAgeTrain)
rand_test = np.random.randint(0 if averageAgeTest - stdAgeTest < 0 else averageAgeTest - stdAgeTest, averageAgeTest + stdAgeTest, N_NanAgeTest)

data_train.loc[np.isnan(data_train["Age"]), "Age"] = rand_train
data_test.loc[np.isnan(data_test["Age"]), "Age"] = rand_test
data_test = data_test.dropna()


def setChildAge(age):
    return 1 if age <= 15 else 0

data_train["Age"] = data_train["Age"].apply(setChildAge)
data_test["Age"] = data_test["Age"].apply(setChildAge)


sc = StandardScaler()
age_scale = sc.fit(data_train['Fare'])
data_train['Fare'] = sc.transform(data_train['Fare'], age_scale)
data_test['Fare'] = sc.transform(data_test['Fare'], age_scale)

X = data_train.iloc[:, 1:].values
y = data_train.Survived

# cross validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
lr = LogisticRegression(C=100.0, random_state=1)

lr.fit(X_train, y_train)
# print lr.predict(X_test)
print lr.score(X_test, y_test)
print cross_validation.cross_val_score(lr, X, y, cv=5)

# ----------------------------------- Logistic_regression_for_test.csv ----------------------------------------------------------- #

test = pd.read_csv('test.csv')
print test.shape
test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

# deal with Sex
test['Sex'] = test['Sex'].map({'male': 1, 'female': 0}).astype(int)

# deal with age
test_mean = test['Age'].mean()
test_std = test['Age'].std()
N_NanAge = test['Age'].isnull().sum()

# generate random numbers
randX = np.random.randint(test_mean - test_std, test_mean + test_std, N_NanAge)
test.loc[np.isnan(test["Age"]), "Age"] = randX
test['Age'] = test['Age'].astype(int)
test['Age'] = test['Age'].apply(setChildAge)

# standardscale for Fare, there's one Nan in Fare item
Ave_Fare = test['Fare'].mean()
Std_Fare = test['Fare'].std()
N_NanFare = test['Fare'].isnull().sum()
randFare = np.random.randint(0 if Ave_Fare - Std_Fare < 0 else Ave_Fare - Std_Fare, Ave_Fare + Std_Fare, N_NanFare)
test.loc[np.isnan(test['Fare']), "Fare"] = randFare

sc = StandardScaler()
age_scale_test = sc.fit(test['Fare'])
test['Fare'] = sc.transform(test['Fare'], age_scale)

# deal with Pclass
dummy_test = pd.get_dummies(test['Pclass'])
dummy_test.columns = ['Pclass1', 'Pclass2', 'Pclass3']
test.drop('Pclass', axis=1, inplace=True)
test = test.join(dummy_test)
ToTest = test.iloc[:, 1:]

# OK, let's predict
# predictions = lr.predict(ToTest)
# print len(predictions)
# print len(test['PassengerId'])
# result = pd.DataFrame({'PassengerId': test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
# result.to_csv("/Users/wei/Documents/data_sci/kaggle/titanic2/LR_predictions.csv", index=False)


# try emsemble-bagging algorithm
# from sklearn.ensemble import BaggingRegressor
# clf = LogisticRegression(C=1000.0, random_state=1)
# bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=False)
# bagging_clf.fit(X, y)
# print bagging_clf.score(X, y)
#
# prediction = bagging_clf.predict(ToTest)

