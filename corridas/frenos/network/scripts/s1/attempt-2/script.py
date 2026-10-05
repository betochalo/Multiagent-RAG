import json
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
acc = GaussianNB().fit(Xtr, ytr).score(Xte, yte)
json.dump({"accuracy_nb": acc}, open("resultados.json", "w"), indent=2)
print(acc)
