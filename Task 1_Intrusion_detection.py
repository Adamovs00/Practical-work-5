import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score,classification_report

df=pd.read_csv('TestbedThuJun17Flows.csv')
AppCount=pd.value_counts(df['appName'])
print(AppCount)
AttackCount=pd.value_counts(df['Label'])
print(AttackCount)
AttackDataFrame=pd.DataFrame(df.loc[df['Label']=='Attack'])
NormalDataFrame=pd.DataFrame(df.loc[df['Label']=='Normal'])
NormalDataFrameY=NormalDataFrame[['Label']]
AttackDataFrameY=AttackDataFrame[['Label']]
AttackDataFrame=AttackDataFrame[['totalSourceBytes','totalDestinationBytes','totalDestinationPackets','totalSourcePackets','sourcePort','destinationPort']]
NormalDataFrame=NormalDataFrame[['totalSourceBytes','totalDestinationBytes','totalDestinationPackets','totalSourcePackets','sourcePort','destinationPort']]

NormalDataFrameY=NormalDataFrameY.to_numpy()
NormalDataFrameY=NormalDataFrameY.ravel()
labels,uniques=pd.factorize(NormalDataFrameY)
NormalDataFrameY=labels
NormalDataFrameY=NormalDataFrameY.ravel()

AttackDataFrameY=AttackDataFrameY.to_numpy()
AttackDataFrameY=AttackDataFrameY.ravel()
labels,uniques=pd.factorize(AttackDataFrameY)
AttackDataFrameY=labels
AttackDataFrameY=AttackDataFrameY.ravel()
indices_zero=AttackDataFrameY==0
AttackDataFrameY[indices_zero]=1

x_train_N, x_test_N, y_train_N, y_test_N = train_test_split(NormalDataFrame, NormalDataFrameY, test_size=80000, random_state=0)
x_train_A, x_test_A, y_train_A, y_test_A = train_test_split(AttackDataFrame, AttackDataFrameY, test_size=5000, random_state=0)

X_train=pd.concat([x_train_N,x_train_A])
X_train=X_train.sample(frac=1, random_state=42)
X_test=pd.concat([x_test_N,x_test_A])
X_test=X_test.sample(frac=1, random_state=42)
Y_train=np.concatenate([y_train_N,y_train_A])
Y_train=pd.DataFrame(Y_train)
Y_train=Y_train.sample(frac=1, random_state=42)
Y_test=np.concatenate([y_test_N,y_test_A])
Y_test=pd.DataFrame(Y_test)
Y_test=Y_test.sample(frac=1, random_state=42)

#clf=DecisionTreeClassifier(random_state=0)
clf=KNeighborsClassifier(n_neighbors=5)#Testing another ML algorithm
clf.fit(X_train,Y_train)
cv=KFold(n_splits=10,random_state=0,shuffle=True)
accuracy=clf.score(X_test,Y_test)
KFold_accuracy=cross_val_score(clf,X_train,Y_train,scoring='accuracy',cv=cv,n_jobs=-1)
print('Accuracy: ',accuracy)
print('KFold Accuracy: ',KFold_accuracy)
predict=clf.predict(X_test)
cm=confusion_matrix(Y_test,predict)
precision=precision_score(Y_test,predict,average='weighted',labels=np.unique(predict))
recall=recall_score(Y_test,predict,average='weighted',labels=np.unique(predict))
f1=f1_score(Y_test,predict,average='macro',labels=np.unique(predict))
print(classification_report(Y_test,predict,target_names=['Normal','Attacks']))
print('Confusion matrix: ',cm)

X_test=X_test.to_numpy()
Y_test=Y_test.to_numpy()
predict=np.reshape(predict,(85000,1))
fp_rows=[]
for i in range(len(predict)):
  if predict[i]==1 and Y_test[i]==0:
    fp_rows.append(i)
print('Number of false alerts: ',len(fp_rows))