'''
Load model from disk

'''

import pickle
import numpy as np

filename="cloth_cat.sav"
cloth_model = pickle.load(open(filename, 'rb'))

def run(image_file, cloth_model):
    
    ''' 
    To change dimension of image, adding channel
    Input should be of feautre size ( 28,28 ) 
    '''

    cloth_category_list = ['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle boot']

    img = (np.expand_dims(image_file,0)) 

    cloth_category_num = np.argmax(cloth_model.predict(img))

    return cloth_category_list[cloth_category_num]


import numpy as np
img = np.load("sample_test_images.npy")


sample = img[0]
print(sample.shape)
print(run(sample,cloth_model))