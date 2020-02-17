from ID3 import ID3DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report, confusion_matrix
from toy import ToyData

attributes, classes, data, target, data2, target2 = ToyData().get_data()

# digits = load_digits()

# digitsLength = digits.data.shape[0]
# trainLength = int(0.7*digitsLength)

# data = digits.data[:trainLength]
# target = digits.target[:trainLength]

# data2 = digits.data[trainLength:]
# testY = digits.target[trainLength:]

id3 = ID3DecisionTreeClassifier()

myTree = id3.fit(data, target, attributes)
# print(myTree)
plot = id3.make_dot_data()
plot.render("testTree")
# predicted = id3.predict(data2, myTree)
# print(predicted)


# id3 = ID3DecisionTreeClassifier()
# root = id3.fit(trainX, trainY)

# predY = id3.predict(testX, root)

# print(predY)
# print(testY)

# print(confusion_matrix(testY, predY))
# print(classification_report(testY, predY))
