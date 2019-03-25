#https://www.kaggle.com/javapocalypse/breast-cancer-classification-in-keras-using-ann
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import metrics


def dataSet():

    dataset = pd.read_csv('G:/ANALYTICS_WORLD_R_SAS/python_world/pycharm projects/ANN/data.csv')
    X = dataset.iloc[:,2:32]
    y = dataset.iloc[:,1]

    return dataset,X,y

dataset,X,y=dataSet()


def dataSetAnalysis(df):
    #view starting values of data set
    print("Dataset Head")
    print(df.head(3))
    print("=" * 30)

    # View features in data set
    print("Dataset Features")
    print(df.columns.values)
    print("=" * 30)

    # View How many samples and how many missing values for each feature
    print("Dataset Features Details")
    print(df.info())
    print("=" * 30)

    # view distribution of numerical features across the data set
    print("Dataset Numerical Features")
    print(df.describe())
    print("=" * 30)

    # view distribution of categorical features across the data set
    print("Dataset Categorical Features")
    print(df.describe(include=['O']))
    print("=" * 30)

dataSetAnalysis(dataset)


def label_encoding(y):

    lencod= LabelEncoder()
    y = lencod.fit_transform(y)

    return y

y=label_encoding(y)


def train_test_splits(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test=train_test_splits(X,y)


def scaling(X_train,df):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    df = sc.transform(df)

    return X_train,df

X_train,X_test=scaling(X_train,X_test)


#Preparing ANN

def build_classifier(optimizer):

    classifier = Sequential()
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [1, 5],
              'epochs': [100, 120],
              'optimizer': ['adam', 'rmsprop']
              }

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

print("best_parameters :"+str(grid_search.best_params_))
print("best_accuracy :"+str( grid_search.best_score_))

#from keras.models import load_model
#classifier.save('breast_cancer_model.h5') #Save trained ANN
#classifier = load_model('breast_cancer_model.h5')  #Load trained ANN

print(confusion_matrix(grid_search.predict(X_test),y_test))
print(accuracy_score(grid_search.predict(X_test),y_test))
