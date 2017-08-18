import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, RMSprop
import keras.callbacks

import matplotlib.pyplot as plt
import math
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import json

img_width, img_height = 224, 224

def train_top(bottleneck_train, bottleneck_validation, train_data_dir='data', validation_data_dir='validation', optimizer='rmsprop', epochs=50, output='out.h5'):
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_training_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_training_top.filenames)
    num_classes = len(generator_training_top.class_indices)


    # get the class lebels for the training data, in the original order
    train_labels = generator_training_top.classes
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_training_top.class_indices)
    
    generator_validation_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_labels = generator_validation_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)
    
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    
    
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+output, histogram_freq=0, write_graph=True, write_images=True)

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        callbacks=[tbCallBack])

    model.save_weights(output+".h5")

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))
    return history

def plotResult(history):
    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()   


def predict(image_path, top_model_weights_path):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)


    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)
    print(probabilities)
    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]
    return label