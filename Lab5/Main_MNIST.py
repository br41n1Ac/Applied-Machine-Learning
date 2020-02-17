from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from Lab5 import MNIST


def main():
    mnist = MNIST.MNISTData('MNIST_Light/*/*.png')

    train_features, test_features, train_labels, test_labels = mnist.get_data()

    mnist.visualize_random()

    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)
    y_pred = gnb.predict(test_features)

    print("Classification report SKLearn GNB:\n%s\n"
          % (metrics.classification_report(test_labels, y_pred)))
    print("Confusion matrix SKLearn GNB:\n%s" % metrics.confusion_matrix(test_labels, y_pred))

    mnist.visualize_wrong_class(y_pred, 8)


if __name__ == "__main__": main()
