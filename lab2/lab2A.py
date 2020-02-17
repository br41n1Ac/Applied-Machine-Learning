from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix


digits = load_digits()

digitsLength = digits.data.shape[0]
trainLength = int(0.7*digitsLength)

trainX = digits.data[:trainLength]
trainY = digits.target[:trainLength]

testX = digits.data[trainLength:]
testY = digits.target[trainLength:]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainX, trainY)

predY = clf.predict(testX)

print(confusion_matrix(testY, predY))
print(classification_report(testY, predY))

# clf2 = tree.DecisionTreeClassifier(min_samples_leaf=2)
# clf2 = tree.DecisionTreeClassifier(min_samples_split=3)
# clf2 = tree.DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5)
# clf2 = clf2.fit(trainX, trainY)

# predY2 = clf2.predict(testX)

# print(confusion_matrix(testY, predY2))
# print(classification_report(testY, predY2))
