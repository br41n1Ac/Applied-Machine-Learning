from keras import models
from keras import layers
import numpy as np
import pickle
from keras.applications import VGG16
from keras_preprocessing.image import ImageDataGenerator
import os

from sklearn import metrics
from sklearn.utils import shuffle

vilde = False
if vilde:
    BASE_DIR = '/home/pierre/Cours/EDAN20/corpus/CoNLL2003/'
else:
    BASE_DIR = '/Users/simonakesson/PycharmProjects/EDAN95Assignment1/Lab3/'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))
conv_base.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy')




def load_conll2003_en():
    train_file = BASE_DIR + 'datasets/flowers_split/train'
    dev_file = BASE_DIR + 'datasets/flowers_split/validation'
    test_file = BASE_DIR + 'datasets/flowers_split/test'
    column_names = ['filename', 'class']
    train_files = shuffle(get_files(train_file, 'jpg'))
    validation_files = get_files(dev_file, 'jpg')
    test_files = get_files(test_file, 'jpg')
    return train_files, validation_files, test_files, column_names


def get_files(dir, suffix):
    """
    Returns all the files in a folder ending with suffix
    :param dir:
    :param suffix:
    :return: the list of file names
    """
    files = []
    for file in os.listdir(dir):
        if not file.startswith('.'):
            for file1 in os.listdir(dir + '/' + file):
                if file1.endswith(suffix):
                    temp_tuple = (file1, file)
                    files.append(temp_tuple)
    return files


train_files, validation_files, test_files, column_names = load_conll2003_en()


train_generator = train_datagen.flow_from_directory(
    directory=BASE_DIR + 'datasets/flowers_split/train',
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


valid_generator = val_datagen.flow_from_directory(
    directory=BASE_DIR + 'datasets/flowers_split/validation',
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=BASE_DIR + 'datasets/flowers_split/test',
    target_size=(150, 150),
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size


try:
    model = pickle.load(open("modelp2adam.p", "rb"))
except FileNotFoundError:
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=valid_generator,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=2
                        )
    pickle.dump(model, open("modelp2adam.p", "wb"))

print('evaluating...')
model.evaluate_generator(generator=valid_generator,
steps=STEP_SIZE_VALID)

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
test_generator.reset()
print('predicting...')
pred=model.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


file_names = test_generator.filenames
classification = []
for files in file_names:
    temp_class, image_name = files.split('/')
    classification.append(train_generator.class_indices[temp_class])


print("Classification report for classifier %s:\n%s\n" % ([], metrics.classification_report(classification, predicted_class_indices)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(classification, predicted_class_indices))