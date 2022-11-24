import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

train = pd.read_excel("data-latih-kmeans.xlsx")
train = train.drop(['no','nama'], axis=1)
train = train.drop([21,22,23,24,25,26], axis=0)
#define X_train & y_train
x_train = train.drop(['kategori'], axis=1)
y_train = train['kategori']

#import data test
test = pd.read_excel('data-uji-kmeans.xlsx')
test = test.drop(['no','nama'], axis=1)
test = test.drop([19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34], axis=0)
#define x_test & y_train
x_test = test.drop(['kategori'], axis=1)
y_test = test['kategori']

# #Normalized data
# #data train
# scaler = RobustScaler()
# normalized_dat = scaler.fit_transform(x_train)
# x_train = pd.DataFrame(normalized_dat, columns=x_train.columns)
# #data test
# normalized_dat1 = scaler.fit_transform(x_test)

# x_test = pd.DataFrame(normalized_dat1, columns=x_test.columns)


#train the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
svm = SVC(random_state=0, C=1, gamma=1.2915496650148839, kernel='rbf')
svm.fit(x_train,y_train)
y_pred = svm.predict(x_test)
print(accuracy_score(y_test,y_pred))
#save the model in pickle format
import pickle 
pickle.dump(svm,open('model.pkl','wb'))