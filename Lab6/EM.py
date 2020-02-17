import numpy as np
import scipy
from scipy.stats import multivariate_normal
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from Lab6 import data
from Lab6.GBC import predict


def get_class_prob(train_labels):
    unique, counts = np.unique(train_labels, return_counts=True)
    probabilties = []
    for nbr in counts:
        probabilties.append(nbr / sum(counts))
    return probabilties


def initialize(train_features, train_labels):
    shape = np.array(train_features).shape
    initial = np.random.uniform(low=0.01, high=0.99, size=(np.unique(train_labels).shape[0], shape[1]))
    std = np.random.uniform(low=0.01, high=0.99, size=(np.unique(train_labels).shape[0], shape[1]))
    pi = get_class_prob(train_labels)
    return pi, initial, std


def get_columns(train_features):
    columns = np.array([train_features[:, i] for i in range(len(train_features[0]))])
    return columns


def e_step_2(train_features, pi, means, variance):
    responsibilities = dict()
    for image_idx in range(len(train_features)):
        temp_image = np.true_divide(train_features[image_idx], 16)
        responsibilities[image_idx] = dict()
        prob = list()
        for k in range(len(pi)):
            # prob.append(pi[k]*get_prob(temp_image, means[k], variance[k]))
            prob.append(pi[k] * scipy.stats.multivariate_normal.pdf(temp_image, mean=means[k], cov=variance[k],
                                                                    allow_singular=True))
        for k in range(len(pi)):
            responsibilities[image_idx][k] = prob[k] / np.sum(np.array(prob))
    return responsibilities


def get_sum_class_r(r, pi):
    sum_class = dict()
    for k in range(len(pi)):
        temp_r = 0
        for i in range(len(r)):
            temp_r += r[i][k]
        sum_class[k] = temp_r
    return sum_class


def m_step(r, pi, data):
    means = []
    variance = []
    sum_class = get_sum_class_r(r, pi)
    for k in range(len(pi)):
        mu_k = 0
        for i in range(len(r)):
            mu_k += r[i][k] * np.true_divide(data[i], 16) / sum_class[k] + 10e-10
        means.append(mu_k)
    means = np.array(means)
    variance = get_variance(r, pi, data, means)
    return means, variance


def get_variance(r, pi, data, means):
    variance = []
    for k in range(len(pi)):
        temp_var = []
        for j in range(len(means[1])):
            temp_sum = 0
            for row in data:
                temp_sum += (row[j] / 16 - means[k][j]) ** 2
            temp_var.append(temp_sum / len(data) + 0.01)
        variance.append(temp_var)
    variance = np.array(variance)
    return variance


def get_prob(x, mu, var):
    temp = (1 / (np.sqrt(2 * np.pi * var)) * np.exp(- (x - mu) ** 2 / (2 * var)))
    return temp


def update_loglikelihood(img, means, cov, pis):
    logs = []
    for image_idx in range(len(img)):
        temp_image = np.true_divide(img[image_idx], 16)
        pdf = np.array(
            [pis[j] * scipy.stats.multivariate_normal.pdf(temp_image, mean=means[j], cov=cov[j], allow_singular=True)
             for j in range(len(pis))])
        log_ll = np.log(np.sum(pdf, axis=0))
        log_ll_sum = np.sum(log_ll)
        logs.append(log_ll_sum)
    logs_sum = np.sum(logs)
    return logs_sum


def main():
    train_features, test_features, train_labels, test_labels = data.digitsData()
    pi, means, covs = initialize(train_features, train_labels)
    i = 0
    likelihood = 0
    new_likelihood = 3
    error = 10e-5
    while abs(likelihood - new_likelihood) > error:
        likelihood = new_likelihood
        r = e_step_2(np.array(train_features), pi, means, covs)
        means, covs = m_step(r, pi, train_features)
        i += 1
        new_likelihood = update_loglikelihood(np.array(train_features), means, covs, pi)
        print(new_likelihood)
        print(i)

    print('class', pi)
    means1 = dict()
    std1 = dict()
    for i in range(len(pi)):
        means1[i] = means[i]
        std1[i] = covs[i]

    labels = np.array(predict(means1, std1, np.true_divide(train_features, 16)))
    idxs = dict()
    new_indices = dict()
    for i in range(len(pi)):
        indices = np.where(labels == i)
        idxs[i] = indices
    for i in range(len(pi)):
        indic = idxs[i][0]
        if len(indic) > 0:
            a = np.array(np.take(train_labels, idxs[i]))
            values, counts = np.unique(a, return_counts=True)
            ind = np.argmax(counts)
            ind = values[ind]
        new_indices[ind] = indic

    new = np.zeros(len(train_features))
    for key in new_indices:
        for ind in new_indices[key]:
            new[ind] = key

    print("Classification report nearest centroid classifier:\n%s\n"
          % (metrics.classification_report(train_labels, new)))
    print("Confusion matrix Nearest centroid classifier:\n%s" % metrics.confusion_matrix(train_labels, new))

    print('completeness score', completeness_score(labels, train_labels))
    print('homogeneity score', "%.6f" % homogeneity_score(labels, train_labels))
    kmeans = KMeans(n_clusters=10, random_state=0).fit(train_features)
    kmeans.labels_
    k_labels = kmeans.predict(train_features)
    print('kmeans completeness score', completeness_score(k_labels, train_labels))
    print('kmeans homogeneity score', "%.6f" % homogeneity_score(k_labels, train_labels))


if __name__ == "__main__": main()
