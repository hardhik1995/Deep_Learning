import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn import metrics


dataset=pd.read_csv("G:/ANALYTICS_WORLD_R_SAS/python_world/deep learning/Artificial_Neural_Networks/Churn_Modelling.csv ")

x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

# Encoding categorical data

LabelEncoder_x_1=LabelEncoder()
x[:,1]=LabelEncoder_x_1.fit_transform(x[:,1])
LabelEncoder_x_2=LabelEncoder()
x[:,2]=LabelEncoder_x_2.fit_transform(x[:,2])

onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

#split the dataset to train & test set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
y_train=np.transpose(y_train)



#Initializing the ANN
classifier=Sequential()

#adding the input layer & 1st hidden layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))

#adding the second input layer
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#adding the output layer
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#compile the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting with ANN & training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

y_pred=classifier.predict(x_test)

y_pred=(y_pred > 0.5)


cm=confusion_matrix(y_test,y_pred)
print(cm)

print("test accuracy :",metrics.accuracy_score(y_test,y_pred))

############################

new_pred=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

new_pred=(new_pred > 0.5) ##customer doesnt leave the bank

###########################
#evaluating , improving ,tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import keras
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=-1)

mean=accuracy.mean()
variance=accuracy.std()

###### Tuning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
            'nb_epoch':[100,500],
            'optimizer':['adam','rmsprop']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

grid_search=grid_search.fit(x_train,y_train)

best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_







