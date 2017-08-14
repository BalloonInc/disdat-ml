
# coding: utf-8

# In[48]:

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import math
import cv2

bottleneck=True
show_history=False
prediction=False

img_width, img_height = 224,224
img_width, img_height = 299,299

top_model_weights_path = 'multiclass-50epochs-inception.h5'
train_data_dir = '../data'
validation_data_dir = '../validation'

# number of epochs to train top model
epochs =  50
# batch size used by flow_from_directory and predict_generator
batch_size = 16


# # Saving bottleneck features

# build the VGG16 network
model = InceptionV3(include_top=False, weights='imagenet')

datagen = ImageDataGenerator(rescale=1. / 255)

# In[5]:

bottleneckValidationGenerator = datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

print(len(bottleneckValidationGenerator.filenames))
print(bottleneckValidationGenerator.class_indices)
print(len(bottleneckValidationGenerator.class_indices))

nb_validation_samples = len(bottleneckValidationGenerator.filenames)

predict_size_validation = int(
    math.ceil(nb_validation_samples / batch_size))



if bottleneck:
    bottleneck_features_validation = model.predict_generator(
        bottleneckValidationGenerator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


bottleneckTrainGenerator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)

nb_train_samples = len(bottleneckTrainGenerator.filenames)
num_classes = len(bottleneckTrainGenerator.class_indices)

predict_size_train = int(math.ceil(nb_train_samples / batch_size))

if bottleneck:
    bottleneck_features_train = model.predict_generator(
        bottleneckTrainGenerator, predict_size_train)

    np.save('bottleneck_features_inception_train.npy',
            bottleneck_features_validation)

# # Training top model
# 

# data prep
train_datagen =  ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size
)

nb_train_samples = len(train_generator.filenames)
num_classes = len(train_generator.class_indices)


# get the class lebels for the training data, in the original order
train_labels = train_generator.classes

# https://github.com/fchollet/keras/issues/3467
# convert the training labels to categorical vectors
train_labels = to_categorical(train_labels, num_classes=num_classes)

# save the class indices to use use later in predictions
np.save('class_inception_indices.npy', train_generator.class_indices)

nb_validation_samples = len(validation_generator.filenames)

validation_labels = validation_generator.classes
validation_labels = to_categorical(
    validation_labels, num_classes=num_classes)


# In[9]:

# load the bottleneck features saved earlier
train_data = np.load('bottleneck_features_inception_train.npy')
validation_data = np.load('bottleneck_features_inception_validation.npy')


# In[49]:

model = Sequential()
model.add(Flatten(input_shape=train_data.shape[1:]))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(optimizer=SGD(lr=0.00002, momentum=0.9),
              loss='categorical_crossentropy', metrics=['accuracy'])


# In[ ]:

history = model.fit(train_data, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(validation_data, validation_labels))

model.save_weights(top_model_weights_path)

(eval_loss, eval_accuracy) = model.evaluate(
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


if show_history:

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


if prediction:
    # # Prediction

    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    image_path = 'path/to/your/test_image'

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

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    print("Image ID: {}, Label: {}".format(inID, label))


    # In[ ]:

    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)

