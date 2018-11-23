'''
Script para entrenar MobilenetFCN8s en linea de comando
'''

import argparse
import os
import pandas as pd
import numpy as np


def run(args):
    lr = args.lr
    epochs = args.epochs
    decay = args.decay
    momentum = args.momentum
    h5file = args.model
    test_set_path = args.test
    hist = args.hist
    dataset = pd.read_csv(os.path.join('/home','wvillegas','dataset-mask', 'full_masks.csv'))

    from utils_fcn import DataGeneratorMobileNet
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(dataset['orig'],dataset['mask'],
                                                        test_size = 0.2, random_state=1)
    partition = {'train':list(X_train),
                'test': list(X_test)}
    img_list = list(X_train) + list(X_test)
    mask_list = list(Y_train) + list(Y_test)
    labels = dict(zip(img_list, mask_list))

    img_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'images_resize')
    masks_path = os.path.join('/home','wvillegas','dataset-mask','dataset_resize', 'masks_resize')

    batch_size = 1

    train_generator = DataGeneratorMobileNet(batch_size=batch_size,img_path=img_path,
                                    labels=labels,list_IDs=partition['train'],n_channels=3,
                                    n_channels_label=1,shuffle=True,mask_path=masks_path)
    valid_generator = DataGeneratorMobileNet(batch_size=batch_size,img_path=img_path,
                                    labels=labels,list_IDs=partition['test'],n_channels=3,
                                    n_channels_label=1,shuffle=True,mask_path=masks_path)
    from keras.applications import MobileNet
    from keras.layers import Conv2DTranspose,Conv2D,Add
    from keras import Model
    net = MobileNet(include_top=False, weights=None)
    net.load_weights('/home/wvillegas/DLProjects/BudClassifier/cmdscripts/modelosV2/mobilenet_weights_detection.h5', by_name=True)

    for layer in net.layers:
        layer.trainable = True

    predict = Conv2D(filters=1,kernel_size=1,strides=1)(net.output)

    deconv2 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same')(predict)

    pred_conv_dw_11_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_dw_11_relu').output)

    fuse1 = Add()([deconv2, pred_conv_dw_11_relu])

    pred_conv_pw_5_relu = Conv2D(filters=1,kernel_size=1,strides=1)(net.get_layer('conv_pw_5_relu').output)

    deconv2fuse1 = Conv2DTranspose(filters=1,kernel_size=4,strides=2, padding='same')(fuse1)

    fuse2 = Add()([deconv2fuse1, pred_conv_pw_5_relu])

    deconv8 = Conv2DTranspose(filters=1,kernel_size=16,strides=8, padding='same')(fuse2)
    fcn = Model(inputs=net.input,outputs=deconv8)
    from keras.optimizers import SGD
    sgd = SGD(lr=lr,momentum=momentum,decay=decay)
    fcn.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = fcn.fit_generator(generator=train_generator,validation_data=valid_generator,
                                use_multiprocessing=True,workers=6, epochs=epochs)
    fcn.save(os.path.join(h5file))
    test_csv = pd.DataFrame({'x':X_test,
                        'y': Y_test})
    test_csv.to_csv(test_set_path,header=None)
    test_csv = pd.DataFrame(history)
    test_csv.to_csv(hist.history)


def main():
    parser = argparse.ArgumentParser(description="Train a default MobileNet binary classification model")
    parser.add_argument("-lr",help="absolute path of training csv (imagepath, label)" ,dest="lr", type=float, required=True)
    parser.add_argument("-ep",help="number of epochs to train on" ,dest="epochs", type=int, required=True)
    parser.add_argument('-decay',help="decay" ,dest="decay", type=float, required=True)
    parser.add_argument('-mmt',help="momentum" ,dest="momentum", type=float, required=True)
    parser.add_argument("-out",help="path to output model file (e.g '/home/mod.h5')" ,dest="model", type=str, required=True)
    parser.add_argument("-test",help="path to output test set csv (e.g '/home/test.csv')" ,dest="test", type=str, required=True)
    parser.add_argument("-hist",help="path to output history csv (e.g '/home/hist.csv')" ,dest="hist", type=str, required=True)
    
    parser.set_defaults(func=run)
    args=parser.parse_args()
    args.func(args)


if __name__=="__main__":
    main()
