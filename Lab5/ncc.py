import pickle

from scipy.cluster.hierarchy import centroid
from sklearn import metrics, datasets
import matplotlib.pyplot as plt
from Lab5 import MNIST, data
import numpy as np


def find_centroid(features, labels):
    centroids = dict()
    centroids_idx = dict()
    for i in range(len(features)):
        if labels[i] not in centroids.keys():
            centroids[labels[i]] = dict()
            centroids_idx[labels[i]] = list()
            print()

        for row in range(len(features[i])):
            label = labels[i]
            if row not in centroids[label].keys():
                centroids[label][row] = list()
            centroids[label][row].append(features[i][row])

    for label in centroids.keys():
        temp_list = []
        for row in centroids[label]:
            sample = centroids[label][row]
            mean = np.array(sample).mean()
            temp_list.append(mean)
        centroids_idx[label] = temp_list
    return centroids_idx


def centroid(feature):
    value = 0
    nbr_not_zero = 0.1
    indexes = 0
    for i in range(len(feature)):
        if feature[i] != 0:
            value += feature[i] * i
            indexes += i
            nbr_not_zero += 1
    value = value / nbr_not_zero
    idx = indexes / len(feature)
    return value, idx


def load_data():
    try:
        train_features = pickle.load(open("train_feat.p", "rb"))
        test_features = pickle.load(open("test_feat.p", "rb"))
        train_labels = pickle.load(open("train_lab.p", "rb"))
        test_labels = pickle.load(open("test_lab.p", "rb"))
    except FileNotFoundError:
        mnist = MNIST.MNISTData('MNIST_Light/*/*.png')
        train_features, test_features, train_labels, test_labels = mnist.get_data()
        pickle.dump(train_features, open("train_feat.p", "wb"))
        pickle.dump(test_features, open("test_feat.p", "wb"))
        pickle.dump(train_labels, open("train_lab.p", "wb"))
        pickle.dump(test_labels, open("test_lab.p", "wb"))
    return train_features, test_features, train_labels, test_labels


def predict(model, test_features):
    labels = []
    for i in range(len(test_features)):
        column = test_features[i]
        min = 1000
        for label in model.keys():
            dist = np.linalg.norm(column - model[label])
            if dist < min:
                min = dist
                current_label = label
        labels.append(current_label)
    return labels


def get_accuracy(labels, test_labels):
    hits = 0
    for i in range(len(labels)):
        if labels[i] == test_labels[i]:
            hits += 1
    return hits / len(labels)


def load_digits():
    digits = datasets.load_digits()
    digits.data
    digits.data[0]

    digits.data.shape

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:10]):
        plt.subplot(2, 5, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    num_examples = len(digits.data)
    num_examples

    num_split = int(0.7 * num_examples)
    num_split

    train_features = digits.data[:num_split]
    train_labels = digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]
    return train_features, test_features, train_labels, test_labels


def main():
    # train_features, test_features, train_labels, test_labels = load_data()
    train_features, test_features, train_labels, test_labels = load_digits()
    train_features, test_features, train_labels, test_labels = data.simplifiedDigitsData()
    model = find_centroid(train_features, train_labels)
    labels = predict(model, test_features)
    hitrate = get_accuracy(labels, test_labels)
    print(hitrate)
    print("Classification report nearest centroid classifier:\n%s\n"
          % (metrics.classification_report(test_labels, labels)))
    print("Confusion matrix Nearest centroid classifier:\n%s" % metrics.confusion_matrix(test_labels, labels))


if __name__ == "__main__": main()
