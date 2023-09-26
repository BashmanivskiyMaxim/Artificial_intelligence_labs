import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from io import BytesIO

iris = load_iris()
X, y = iris.data, iris.target
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=0)

clf = RidgeClassifier(tol=1e-2, solver="sag")
clf.fit(Xtrain, ytrain)
ypred = clf.predict(Xtest)

accuracy = np.round(metrics.accuracy_score(ytest, ypred), 4)
precision = np.round(metrics.precision_score(ytest, ypred, average='weighted'), 4)
recall = np.round(metrics.recall_score(ytest, ypred, average='weighted'), 4)
f1_score = np.round(metrics.f1_score(ytest, ypred, average='weighted'), 4)
cohen_kappa = np.round(metrics.cohen_kappa_score(ytest, ypred), 4)
matthews_corrcoef = np.round(metrics.matthews_corrcoef(ytest, ypred), 4)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1_score)
print('Cohen Kappa Score:', cohen_kappa)
print('Matthews Corrcoef:', matthews_corrcoef)

classification_report = metrics.classification_report(ytest, ypred)
print('\t\tClassification Report:\n', classification_report)

mat = confusion_matrix(ytest, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('True label')
plt.ylabel('Predicted label')
plt.savefig("Confusion.jpg")

# Save SVG in a fake file object.
f = BytesIO()
plt.savefig(f, format="svg")
