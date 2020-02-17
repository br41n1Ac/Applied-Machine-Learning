import pickle

import keras
from keras.utils import to_categorical
from sklearn import metrics

keras.__version__
from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))


test_datagen = ImageDataGenerator(rescale=1. / 255)


base_dir = '/Users/simonakesson/PycharmProjects/EDAN95Assignment1/Lab3/'

test_generator = test_datagen.flow_from_directory(
    directory=base_dir + 'datasets/flowers_split/test',
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)

train_dir = os.path.join(base_dir, 'datasets/flowers_split/train')
validation_dir = os.path.join(base_dir, 'datasets/flowers_split/validation')
test_dir = os.path.join(base_dir, 'datasets/flowers_split/test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = np.argmax(labels_batch,axis=1)
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels



try:
    train_features = pickle.load(open("train_features.p", "rb"))
    train_labels = pickle.load(open("train_labels.p", "rb"))
except FileNotFoundError:
    train_features, train_labels = extract_features(train_dir, 2400)
    pickle.dump(train_features, open("train_features.p", "wb"))
    pickle.dump(train_labels, open("train_labels.p", "wb"))


try:
    validation_features = pickle.load(open("valid_features.p", "rb"))
    validation_labels = pickle.load(open("valid_labels.p", "rb"))
except FileNotFoundError:
    validation_features, validation_labels = extract_features(validation_dir, 800)
    pickle.dump(validation_features, open("valid_features.p", "wb"))
    pickle.dump(validation_labels, open("valid_labels.p", "wb"))


try:
    test_features = pickle.load(open("test_features.p", "rb"))
    test_labels = pickle.load(open("test_labels.p", "rb"))
except FileNotFoundError:
    test_features, test_labels = extract_features(test_dir, 800)
    pickle.dump(validation_features, open("test_features.p", "wb"))
    pickle.dump(validation_labels, open("test_labels.p", "wb"))


train_features = np.reshape(train_features, (2400, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (800, 4 * 4 * 512))
test_features = np.reshape(test_features, (800, 4 * 4 * 512))

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features, to_categorical(train_labels),
                    epochs=40,
                    batch_size=20,
                    validation_data=(validation_features, to_categorical(validation_labels)))

pred = model.predict(test_features)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


file_names = test_generator.filenames
classification = []
for files in file_names:
    temp_class, image_name = files.split('/')
    classification.append(test_generator.class_indices[temp_class])


print("Classification report for classifier %s:\n%s\n" % ([], metrics.classification_report(test_labels, predicted_class_indices)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, predicted_class_indices))