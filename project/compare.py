import smote
import rus
import warnings
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report

def adaboost(X_train, X_test, y_train):
    model = AdaBoostClassifier(learning_rate=0.7,n_estimators=500, random_state=42)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test) 
    return y_pred

warnings.filterwarnings("ignore")
train_data = pd.read_csv("Dataset/train_data.csv")
test_data = pd.read_csv("Dataset/test_data.csv")
train_label = train_data["FaultCause"]
train_data = train_data.drop(["FaultCause"], axis=1)
test_label = test_data["FaultCause"]
test_data = test_data.drop(["FaultCause"], axis=1)
all_data = pd.concat([train_data, test_data], axis=0)
all_label = pd.concat([train_label, test_label], axis=0)
all_data = all_data.values
all_label = all_label.values

#获取数据并将其分为训练集合测试集
X_train, X_test, y_train, y_test = train_test_split(all_data, all_label, test_size=0.3)

#使用SMOTE算法调整训练集的分布，为小样本类添加新样本
all_resampled_data, all_resampled_label = SMOTE().fit_resample(X_train, y_train)

#使用Adaboost算法对调整分布后的数据进行分类
y_baseline = adaboost(all_resampled_data, X_test, all_resampled_label)

#KNN
clf = neighbors.KNeighborsClassifier(n_neighbors = 15 , weights='distance')
clf.fit(X_train, y_train) 
knn_pred = clf.predict(X_test)

#SVM
classiff = svm.SVC(kernel='rbf')
clff = classiff.fit(X_train, y_train)
svm_pred = clff.predict(X_test)

#Adaboost
y_baseline = adaboost(X_train, X_test, y_train)

 


print('KNN')
print('############################################') 
print(classification_report(y_test, knn_pred)) 
print('############################################')
print('SVM')
print('############################################') 
print(classification_report(y_test, svm_pred)) 
print('############################################')
print('adaboost')
print('############################################') 
print(classification_report(y_test, y_baseline)) 
print('############################################')
print('SVMscore')
print('############################################') 
print(accuracy_score(y_test, svm_pred)) 
print('############################################')  

#计算最后的分类精度
print(accuracy_score(y_test, y_baseline)) 