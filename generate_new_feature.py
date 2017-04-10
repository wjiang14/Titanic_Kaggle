import pandas as pd
import numpy as np
from API.API import cleanTicket
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

full = data_train.append(data_test, ignore_index = True)
titanic = full[:891]
len_train = 891
del data_train, data_test

# print 'DataSets:', 'full:', full.shape, 'titanic:', titanic.shape
# print titanic.corr()

sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name='Sex')
embarked = pd.get_dummies(full.Embarked, prefix='Embared')
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')

# --------------------------------------------------------#
# fill missing values in variables
imputed = pd.DataFrame()
# fill mising values with average of Fare
age_mean = full['Age'].mean()
age_std = full['Age'].std()
N_nonAge = full['Age'].isnull().sum()

rand_train = np.random.randint(0 if age_mean - age_std < 0 else age_mean - age_std, age_mean + age_std, N_nonAge)
imputed['Age'] = full['Age']
imputed.loc[np.isnan(imputed["Age"]), "Age"] = rand_train
#imputed['Age'] = full.Age.fillna(full.Age.mean())
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())


# Feature engineer: generate new feature
title = pd.DataFrame()
title['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())
title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess": "Royalty",
    "Dona": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty"
}

title['Title'] = title['Title'].map(title_Dictionary)
title = pd.get_dummies(title['Title'])


# --------------------------------------------------------#
# Extract Cabin category information
cabin = pd.DataFrame()
cabin['Cabin'] = full.Cabin.fillna('U')
# mapping cabin value with the cabin letter
cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])
# encoding
cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin_')

# --------------------------------------------------------#
# Deal with ticket
ticket = pd.DataFrame()
ticket['Ticket'] = full['Ticket'].map(cleanTicket)
ticket = pd.get_dummies(ticket['Ticket'], prefix='Ticket')
# print ticket.shape
# print ticket.head()


# --------------------------------------------------------#
# creat family size and category for family size
family = pd.DataFrame()
# introducing a new feature: the size pf families (including the passenger)
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

# introducing other features based on the family size,
family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s == 1 else 0)
family['Family_Small'] = family['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
family['Family_Large'] = family['FamilySize'].map(lambda s: 1 if 5 <= s else 0)
# print family


# --------------------------------------------------------#
# Select with features/variables to include in the dataset from the list below:
full_X = pd.concat([imputed, embarked, cabin, sex], axis=1)


# --------------------------------------------------------#
# train data set
train_valid_X = full_X[:891]
train_valid_y = titanic.Survived
test_X = full_X[891:]
train_X, valid_X, train_y, valid_y = train_test_split(train_valid_X, train_valid_y, test_size=0.3, random_state=0)
print full_X.shape, train_X.shape, valid_X.shape, train_y.shape, valid_y.shape, test_X.shape
# print plot_variable_importance(train_X, train_y)

# --------------------------------------------------------#
# train data set
# Model selection
# model = LogisticRegression()
# model = RandomForestClassifier(n_estimators=100)
# model = SVC()
model = GradientBoostingClassifier()
model.fit(train_X, train_y)

# evaluation
print model.score(train_X, train_y)

# Automagic
rfecv = RFECV( estimator = model , step = 1 , cv = StratifiedKFold( train_y , 2 ) , scoring = 'accuracy' )
rfecv.fit( train_X , train_y )

print rfecv.score( train_X , train_y ) , rfecv.score( valid_X , valid_y )
print "Optimal number of features : %d" % rfecv.n_features_

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel( "Number of features selected" )
plt.ylabel( "Cross validation score (nb of correct classifications)" )
plt.plot( range( 1 , len( rfecv.grid_scores_ ) + 1 ) , rfecv.grid_scores_ )
plt.show()

test_Y = model.predict( test_X ).astype(int)
passenger_id = full[891:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': test_Y } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )