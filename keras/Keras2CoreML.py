from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.layers import Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.models import load_model
from PIL import Image as IMG
import numpy as np
import coremltools
import os

model1 = load_model('disdat-keras.model')
#model2 = load_model('../disdat-v1.model')

elephant_pic = 'elephant.jpg'
peacock_pic = 'peacock.jpg'

elephant_img = IMG.open(elephant_pic)
peacock_img = IMG.open(peacock_pic)
elephant = image.img_to_array(elephant_img)
peacock = image.img_to_array(peacock_img)
elephant = np.expand_dims(elephant, axis=0)
peacock = np.expand_dims(peacock, axis=0)
elephant = preprocess_input(elephant)
peacock = preprocess_input(peacock)

elephant_preds1 = model1.predict(elephant)
peacock_preds1 = model1.predict(peacock)

#elephant_preds2 = model2.predict(elephant)
#peacock_preds2 = model2.predict(peacock)

print("KERAS - model 1")
print('Elephant Probabilities:\n', elephant_preds1)
print('Peacock Probabilities:\n', peacock_preds1)

# print("KERAS - model 2")
# print('Elephant Probabilities:\n', decode_predictions(elephant_preds2, top=3))
# print('Peacock Probabilities:\n', decode_predictions(peacock_preds2, top=3))

exit()

coreml_model = coremltools.converters.keras.convert(model,
                                                    input_names=['image'],
                                                    output_names=['probabilities'],
                                                    image_input_names='image',
                                                    class_labels='labels.txt',
                                                    predicted_feature_name='class')

coreml_model.save('DisDatv1.mlmodel')

print("Converted model saved.")