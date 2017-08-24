import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.vgg16 import VGG16, preprocess_input

from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD, RMSprop
import keras.callbacks

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import math
import operator
import cv2

img_width, img_height = 224, 224

def version():
    return "1.0"

def bottleneck(model,data_dir,output, batch_size):
    datagen = ImageDataGenerator(rescale=1. / 255)

    bottleneckGenerator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_samples = len(bottleneckGenerator.filenames)

    predict_size = int(math.ceil(nb_samples / batch_size))

    bottleneck_features = model.predict_generator(bottleneckGenerator, predict_size)

    np.save(output, bottleneck_features)
    return bottleneck_features


def train_top(train_data, validation_data, train_data_dir='data', validation_data_dir='validation', optimizer='rmsprop', batch_size=32, epochs=50, output='out.h5'):
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
    model.add(Dense(num_classes, activation='softmax'))

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
    return history, model

def finetune(train_data_dir='data', validation_data_dir='validation', optimizer='rmsprop', weights_top_layer='out.h5', batch_size=32, epochs=5, output='out-refined'):
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3))

    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_training_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(generator_training_top.class_indices)
    
    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(num_classes, activation='softmax'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(weights_top_layer)

    # add the model on top of the convolutional base
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    
    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='categorical')

    nb_train_samples = len(train_generator.filenames)

    nb_validation_samples = len(validation_generator.filenames)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/'+output, histogram_freq=0, write_graph=True, write_images=True)

    saveCallBack = keras.callbacks.ModelCheckpoint(output+'-ep{epoch:02d}-valacc{val_acc:.2f}.h5', monitor='val_acc', save_best_only=True, save_weights_only=True)

    # fine-tune the model
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size,
            callbacks=[tbCallBack, saveCallBack])

    model.save_weights(output+".h5")
    return history, model
    
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


def predict_top(image_path, top_model_weights_path, class_dictionary):
    num_classes = len(class_dictionary)

    orig = cv2.imread(image_path)

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
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)


    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    prediction = model.predict_proba(bottleneck_prediction)

    predictions = {list(class_dictionary.keys())[list(class_dictionary.values()).index(idx)]:val for (idx, val) in enumerate(prediction[0])}
    sorted_predictions = sorted(predictions.items(), key=operator.itemgetter(1))
    sorted_predictions.reverse()
    return sorted_predictions[0:4]

def predict_fine(model, class_dictionary, img_path='test_images/keyboard.jpg'):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predictions = {list(class_dictionary.keys())[list(class_dictionary.values()).index(idx)]:val for (idx, val) in enumerate(prediction[0])}
    sorted_predictions = sorted(predictions.items(), key=operator.itemgetter(1))
    sorted_predictions.reverse()
    return sorted_predictions[0:4]