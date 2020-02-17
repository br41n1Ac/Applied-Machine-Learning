from sklearn import datasets, svm, metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

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


num_split = int(0.7*num_examples)
num_split


train_features = digits.data[:num_split]
train_labels = digits.target[:num_split]
test_features = digits.data[num_split:]
test_labels = digits.target[num_split:]

print("Number of training examples: ", len(train_features))
print("Number of test examples: ", len(test_features))
print("Number of total examples:", len(train_features)+len(test_features))

classifier = DecisionTreeClassifier(random_state=0,max_leaf_nodes=1000)

classifier.fit(train_features, train_labels)

predicted = classifier.predict(test_features)
print(predicted)
predicted


print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(test_labels, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted))

print('train feat',train_features)
print('train lab', train_labels)