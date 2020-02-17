from sklearn.datasets import load_digits


def digitsData():
    features = load_digits().data
    labels = load_digits().target

    length = features.shape[0]
    length70 = int(0.7 * length)

    trainFeatures = features[:length70]
    trainLabels = labels[:length70]

    testFeatures = features[length70:]
    testLabels = labels[length70:]

    return trainFeatures, testFeatures, trainLabels, testLabels


def simplifiedDigitsData():
    features = load_digits().data
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            if features[i][j] < 5:
                features[i][j] = 0
            elif features[i][j] < 11:
                features[i][j] = 1
            else:
                features[i][j] = 2

    labels = load_digits().target

    length = features.shape[0]
    length70 = int(0.7 * length)

    trainFeatures = features[:length70]
    trainLabels = labels[:length70]

    testFeatures = features[length70:]
    testLabels = labels[length70:]

    return trainFeatures, testFeatures, trainLabels, testLabels