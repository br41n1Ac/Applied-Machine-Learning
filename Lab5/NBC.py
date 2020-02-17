import math
import pickle
import numpy as np
from sklearn import datasets, metrics
import matplotlib.pyplot as plt

from Lab5 import MNIST, data


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


def find_mean(features, labels):
    pixel_values_classes = dict()
    for i in range(len(features)):
        if labels[i] not in pixel_values_classes.keys():
            pixel_values_classes[labels[i]] = dict()

        for row in range(len(features[i])):
            label = labels[i]
            if row not in pixel_values_classes[label].keys():
                pixel_values_classes[label][row] = list()
            pixel_values_classes[label][row].append(features[i][row])

    temp = dict()
    for classes in pixel_values_classes:
        temp[classes] = dict()
        pixel_values = pixel_values_classes[classes]
        for pixel in pixel_values:
            samples = pixel_values[pixel]
            temp[classes][pixel] = dict()
            uniques, counts = np.unique(samples, return_counts=True)
            pixel_probabilities = dict.fromkeys([*range(0, int(np.max(features)) + 1, 1)])
            for key, value in pixel_probabilities.items():
                if value is None:
                    pixel_probabilities[key] = 0.00001
            for nbr in range(len(counts)):
                pixel_probabilities[uniques[nbr]] = counts[nbr] / sum(counts)
            temp[classes][pixel] = pixel_probabilities
            print()

    return temp


"""
            for sample in pixel_values[label][row]:
                uniques, counts = np.unique(sample, return_counts=True)
                pixel_probabilities = dict.fromkeys([*range(0, int(np.max(features)) + 1, 1)])
                for nbr in range(len(counts)):
                    pixel_probabilities[uniques[nbr]] = counts[nbr]/sum(counts)
                temp_prob_list.append(pixel_probabilities)
"""

def predict2(model, test_features, probs):
    prediction = []

    for x in test_features:
        prob_per_class = []
        for classes in model:
            temp_prob = probs[classes]
            for pixel in model[classes]:
                probabilities = model[classes][pixel]
                temp_prob *= probabilities[x[pixel]]
                print()
            prob_per_class.append(temp_prob)
        prediction.append(np.argmax(prob_per_class))
        print()
    return prediction


def get_class_prob(train_labels):
    unique, counts = np.unique(train_labels, return_counts=True)
    probabilties = []
    for nbr in counts:
        probabilties.append(nbr / sum(counts))
    return probabilties


def main():
    # train_features, test_features, train_labels, test_labels = load_data()
    # train_features, test_features, train_labels, test_labels = load_digits()
    train_features, test_features, train_labels, test_labels = data.simplifiedDigitsData()
    get_class_prob(train_labels)
    model = find_mean(train_features, train_labels)
    probabilities = get_class_prob(train_labels)
    labels = predict2(model, test_features, probabilities)
    print()
    print("Classification report nearest centroid classifier:\n%s\n"
          % (metrics.classification_report(test_labels, labels)))
    print("Confusion matrix Nearest centroid classifier:\n%s" % metrics.confusion_matrix(test_labels, labels))


if __name__ == "__main__": main()
