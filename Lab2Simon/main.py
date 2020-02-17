import ToyData as td
import ID3

import numpy as np
from sklearn import tree, metrics, datasets


def main():

    attributes, classes, data, target, data2, target2 = td.ToyData().get_data()
    id3 = ID3.ID3DecisionTreeClassifier()
    toy = True
    if toy:
        myTree = id3.fit(data, target, attributes)
        print(myTree)
        plot = id3.make_dot_data()
        plot.render("testTree")
        predicted = id3.predict(data2, myTree, toy)
        print(target2)
        print(predicted)
    else:

        digits = datasets.load_digits()
        digits.data
        digits.data.shape

        num_examples = len(digits.data)
        num_examples

        num_split = int(0.7 * num_examples)
        num_split
        train_features = digits.data[:num_split]
        train_labels = digits.target[:num_split]
        digits_attributes = {}

        for i in range(len(train_features[0])):
            digits_attributes[i] = list(range(0, 17))
        myTree = id3.fit(train_features, train_labels, digits_attributes)
        print(myTree)
        plot = id3.make_dot_data()
        plot.render("testTree")
        digits.data.shape

        test_features = digits.data[num_split:]
        test_labels = digits.target[num_split:]
        predicted = np.array(id3.predict(test_features, myTree, toy)).astype(int)
        print(test_labels)
        predicted.reshape(540,)
        print(predicted)
        print("Classification report for classifier %s:\n%s\n"% (myTree, metrics.classification_report(test_labels, predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))


if __name__ == "__main__": main()