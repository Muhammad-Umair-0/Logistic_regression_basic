from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load the following dataset

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.20, random_state=23)
model = LogisticRegression(max_iter = 10000)
model.fit( X_train, y_train)
pred = model.predict(X_test)
print(pred)
acc  = accuracy_score(y_test, pred)
print("The accuracy of the model is : ", acc)