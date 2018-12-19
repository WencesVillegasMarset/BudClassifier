from keras.utils import Sequence
import numpy as np
import os.path as path
# from skimage.io import imread
import cv2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input


class DataGeneratorMobileNet(Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=None,dim_label=None, n_channels=3,
                n_channels_label = 1, shuffle=True,img_path='/',mask_path='/'):
                self.dim = dim
                self.dim_label = dim_label
                self.batch_size = batch_size
                self.labels = labels
                self.n_channels_label = n_channels_label
                self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.shuffle = shuffle
                self.on_epoch_end()
                self.img_path = img_path
                self.mask_path = mask_path
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):

        #Generation of data

        for index, id in enumerate(list_IDs_temp):
            #store image array
            temp = cv2.imread(path.join(self.img_path,id))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)

            X = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels))
            X[index,] = temp
            #store mask array
            temp = cv2.imread(path.join(self.mask_path,self.labels[id]))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)
            y = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels_label))
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.astype(bool).astype(int)
            temp = np.expand_dims(temp, axis=2)
            y[index,] = temp
        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    
    
    
    
    
class DataGeneratorFCN(Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=None,dim_label=None, n_channels=3,
                n_channels_label = 1, shuffle=True,img_path='/',mask_path='/'):
                self.dim = dim
                self.dim_label = dim_label
                self.batch_size = batch_size
                self.labels = labels
                self.n_channels_label = n_channels_label
                self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.shuffle = shuffle
                self.on_epoch_end()
                self.img_path = img_path
                self.mask_path = mask_path
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):

        #Generation of data

        for index, id in enumerate(list_IDs_temp):
            #store image array
            temp = cv2.imread(path.join(self.img_path,id))
            X = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels))
            temp = cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
            X[index,] = temp
            #store mask array
            temp = cv2.imread(path.join(self.mask_path,self.labels[id]))
            y = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels_label))
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.astype(bool).astype(int)
            temp = np.expand_dims(temp, axis=2)
            y[index,] = temp
        return X, y
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
class DataGeneratorMobileNetKeras(Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=None,dim_label=None, n_channels=3,
                n_channels_label = 1, shuffle=True,img_path='/',mask_path='/'):
                self.dim = dim
                self.dim_label = dim_label
                self.batch_size = batch_size
                self.labels = labels
                self.n_channels_label = n_channels_label
                self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.shuffle = shuffle
                self.on_epoch_end()
                self.img_path = img_path
                self.mask_path = mask_path
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):

        #Generation of data

        for index, id in enumerate(list_IDs_temp):
            #store image array
            
            temp = preprocess_input(image.img_to_array(image.load_img(path.join(self.img_path,id))))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)

            X = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels))
            X[index,] = temp
            #store mask array
            
            temp = image.img_to_array(image.load_img(path.join(self.mask_path,self.labels[id]), color_mode = "grayscale"))
            temp = cv2.resize(temp, (0, 0), fx=0.5, fy=0.5)
            y = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels_label))
            temp = temp.astype(bool).astype(int)
            temp = np.expand_dims(temp, axis=2)
            y[index,] = temp
        return X, y

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y    
    
'''    
class DataGeneratorFCN32(Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, dim=None,dim_label=None, n_channels=3,
                n_channels_label = 1, shuffle=True,img_path='/',mask_path='/'):
                self.dim = dim
                self.dim_label = dim_label
                self.batch_size = batch_size
                self.labels = labels
                self.n_channels_label = n_channels_label
                self.list_IDs = list_IDs
                self.n_channels = n_channels
                self.shuffle = shuffle
                self.on_epoch_end()
                self.img_path = img_path
                self.mask_path = mask_path
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __data_generation(self, list_IDs_temp):

        #Generation of data

        for index, id in enumerate(list_IDs_temp):
            #store image array
            temp = cv2.imread(path.join(self.img_path,id))
            temp = cv2.resize(temp, (0,0),fx=0.5,fy=0.5)
            X = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels))
            temp = cv2.normalize(temp, temp, 0, 255, cv2.NORM_MINMAX)
            X[index,] = temp
            #store mask array
            temp = cv2.imread(path.join(self.mask_path,self.labels[id]))
            temp = cv2.resize(temp, (0,0),fx=0.5,fy=0.5)
            if ((temp.shape[0] == 2656) and (temp.shape[1] ==1494)):
                temp = cv2.resize(temp, (1472, 2656)) #hecho fcn-32
            elif ((temp.shape[0] == 1494) and (temp.shape[1] == 2656)): 
                temp = cv2.resize(temp, (2656, 1472)) #hecho fcn-32 1472
            if ((temp.shape[0] == 1728) and (temp.shape[1] ==2304)): 
                temp = cv2.resize(temp, (2304, 1728)) #hecho con fcn-32
            if ((temp.shape[0] == 1824) and (temp.shape[1] ==2736)):
                temp = cv2.resize(temp, (2720, 1824)) #hecho con fcn-32
            #if image is square then output shape is the equal 
            y = np.empty((self.batch_size,temp.shape[0],temp.shape[1], self.n_channels_label))
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp = temp.astype(bool).astype(int)
            temp = np.expand_dims(temp, axis=2)
            y[index,] = temp
        return X, y
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
'''             

