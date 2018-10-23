'''
Script para entrenar Mobilenet en linea de comando
'''

import argparse
import os
import utils
import numpy as np

def run(args):
    train_csv = args.src
    epochs = args.epochs
    cw = args.cw
    bs = args.bs
    h5file = args.model
    #imports
    import pandas as pd
    train = []
    train_label = []
    csv = pd.read_csv(train_csv,header=None)
    print(csv.columns)
    for index, row in csv.iterrows():
        train.append(utils.preprocess(row[0]))
        train_label.append(row[1])
    train = np.asarray(train)
    train_label = np.asarray(train_label)
    print(str(train.shape[0]) + ' images preprocessed!')
    model = utils.build()
    class_weight = {0: 1.,
                    1: cw}
    model.fit(x=train, y=train_label,epochs=epochs, batch_size=bs,class_weight=class_weight,shuffle=True)
    model.save(os.path.join(h5file))

def main():
    parser = argparse.ArgumentParser(description="Train a default MobileNet binary classification model")
    parser.add_argument("-src",help="absolute path of training csv (imagepath, label)" ,dest="src", type=str, required=True)
    parser.add_argument("-ep",help="number of epochs to train on" ,dest="epochs", type=int, required=True)
    parser.add_argument('-bs',help="batch size" ,dest="bs", type=int, required=True)
    parser.add_argument('-cw',help="weight of positive class" ,dest="cw", type=float, required=True)
    parser.add_argument("-out",help="path to output model file (e.g '/home/mod.h5')" ,dest="model", type=str, required=True)
    parser.set_defaults(func=run)
    args=parser.parse_args()
    if (not os.path.exists(args.src)):
        parser.error('Invalid path to source csv')
    args.func(args)


if __name__=="__main__":
	main()
