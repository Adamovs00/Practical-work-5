import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import numpy as np

df = pd.read_csv('SDN_traffic.csv')
y = df['category']
y=y.to_numpy()
X = df.drop(['category','id_flow','nw_src','nw_dst'], axis=1)
X=X.to_numpy()
m, n =X.shape
for i in range(m):
    for j in range(n):
        if str(X[i,j]).isnumeric()==False:
            X[i,j]='0'
X=X.astype(float)
print(X,y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cart_model = DecisionTreeClassifier(random_state=42)
cart_model.fit(X_train, y_train)
cart_predictions = cart_model.predict(X_test)
print("CART Model Performance:")
print(classification_report(y_test, cart_predictions))

id3_model = DecisionTreeClassifier(criterion='entropy', random_state=42) # Entropy can be considered a proxy for ID3's approach
id3_model.fit(X_train, y_train)
id3_predictions = id3_model.predict(X_test)
print("ID3 Model Performance:")
print(classification_report(y_test, id3_predictions))