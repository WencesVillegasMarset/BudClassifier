'''
Modulo de funciones auxiliares para predict.py 
'''

import cv2

def preprocess(path):
    img = cv2.imread(path)
    #TODO checkear que no devuelva NULL y si lo hace manejarlo
    if img.shape[0] > 224 and img.shape[1] > 224: 
        #shrink
        img = cv2.resize(img, (224,224),cv2.INTER_AREA)
    else: #zoom
        img = cv2.resize(img, (224,224),cv2.INTER_LINEAR)
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    return img
def build():
    from keras.applications.mobilenet import MobileNet
    from keras.layers import Dense, Flatten
    from keras.models import Model
    from keras.optimizers import SGD
    mobilenet = MobileNet(weights='imagenet', input_shape=(224,224,3), include_top=False)
    flatten = Flatten()(mobilenet.output)
    fc1 = Dense(1024, activation='relu')(flatten)
    final = Dense(1,activation='sigmoid')(fc1)
    model = Model(mobilenet.input, final)
    optim = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=optim,
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return model
