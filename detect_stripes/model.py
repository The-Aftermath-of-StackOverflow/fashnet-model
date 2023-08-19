def import_modules():
    
    ''' Importing the Keras libraries and packages '''

    import os
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from keras.models import Sequential
    from keras.layers import Conv2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense

    from keras.preprocessing.image import ImageDataGenerator

def compile_model():
    # Initialising the CNN
    classifier = Sequential()

    # Step 1 - Convolution
    classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

    # Step 2 - Pooling
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Adding a second convolutional layer
    classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))

    # Step 3 - Flattening
    classifier.add(Flatten())

    # Step 4 - Full connection
    classifier.add(Dense(units = 128, activation = 'relu'))
    classifier.add(Dense(units = 17, activation = 'softmax'))

    # Compiling the CNN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

def train_model():
    ''' Fit CNN to Image '''

    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    training_set = train_datagen.flow_from_directory('dataset_category',
                                                    target_size = (64, 64),
                                                    batch_size = 32,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory('dataset_category_test',
                                                target_size = (64, 64),
                                                batch_size = 32,
                                                class_mode = 'categorical')


    classifier.summary()

    classifier.fit_generator(training_set,
                            steps_per_epoch = 800,
                            epochs = 5,
                            validation_data = test_set,
                            validation_steps = 200)


    ''' 
    save model to Disk by serialize model to JSON 
    save weights in HDF5 mode
    '''
    
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model.h5")
    print("Saved model to disk")
    


dress_patterns_df = pd.read_csv('dress_patterns.csv')
dress_patterns = dress_patterns_df.values
print(os.listdir('dataset_category'))
print(os.listdir('dataset_category/animal'))

compile_model()
train_model()
print("Train success, run model_eval to get results")