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
    centroids = dict()
    centroids_idx = dict()
    std = dict()
    for i in range(len(features)):
        if labels[i] not in centroids.keys():
            centroids[labels[i]] = dict()
            centroids_idx[labels[i]] = list()
            std[labels[i]] = list()

        for row in range(len(features[i])):
            label = labels[i]
            if row not in centroids[label].keys():
                centroids[label][row] = list()
            centroids[label][row].append(features[i][row])

    for label in centroids.keys():
        temp_list = []
        temp_std_list = []
        for row in centroids[label]:
            sample = centroids[label][row]
            mean = np.array(sample).mean()
            temp_std = np.std(np.array(sample), axis=0)
            temp_list.append(mean)
            temp_std_list.append(temp_std**2+0.01)
        centroids_idx[label] = temp_list
        std[label] = temp_std_list
    return centroids_idx, std

def predict(mean, std, test_features):
    prediction = []
    for x in test_features:
        probabilities = []
        for i in range(len(std)):
            finalprob = 1
            for j in range(len(std[i])):
                temp_std = std[i][j]
                temp_mean = mean[i][j]
                temp_feat = x[j]
                temp_prob = 1 / (np.sqrt(2 * np.pi * temp_std)) * np.exp(- (temp_feat - temp_mean) ** 2 / (2 * temp_std))
                if math.isnan(temp_prob) or temp_prob == 0:
                    temp_prob = 0.001
                finalprob *= temp_prob
            probabilities.append(finalprob)
        prediction.append(np.argmax(probabilities))
    return prediction

def main():
    #train_features, test_features, train_labels, test_labels = load_data()
    train_features, test_features, train_labels, test_labels = load_digits()
    #train_features, test_features, train_labels, test_labels = data.simplifiedDigitsData()
    mean, std = find_mean(train_features, train_labels)
    labels = predict(mean, std, test_features)
    print()
    print("Classification report nearest centroid classifier:\n%s\n"
          % (metrics.classification_report(test_labels, labels)))
    print("Confusion matrix Nearest centroid classifier:\n%s" % metrics.confusion_matrix(test_labels, labels))


if __name__ == "__main__": main()