import json
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
X, y = load_breast_cancer(return_X_y=True)
model = DecisionTreeClassifier(random_state=0).fit(X, y)
acc = (model.predict(X) == y).mean()
json.dump({"accuracy": float(acc)}, open("resultados.json", "w"), indent=2)
print(acc)
