import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import model_from_json
# %matplotlib inline

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

print("Import library success")

def test_model(img_location = 'dataset_category/letter_numb/851505710.jpg'):
    
    ''' 
    input : jpg image of variable size
    output : class of image
    '''

    # ----- LOAD SAVED MODEL -----
    json_file = open('model.json', 'r')     
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk success.")

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
    training_set = train_datagen.flow_from_directory('dataset_category',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

    test_image = image.load_img(img_location, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    # training_set.class_indices

    result[0]
    index = np.where(result[0] == 1)[0][0]
    predicted_category = list(training_set.class_indices.keys())[list(training_set.class_indices.values()).index(index)]
    return predicted_category

print("The preidcted class of the given image is :",test_model())