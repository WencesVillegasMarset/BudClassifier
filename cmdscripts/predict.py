'''
Script para testear prediccion en linea de comando
'''
import argparse
import os
import utils
import numpy as np
def run(args):
    source_path = args.source
    output_csv = args.output 
    model_path = args.model
    
    #imports    
    from keras.models import load_model
    #check if path is valid
    #load model
    from keras.models import load_model
    print('Loading Model...')
    model = load_model(os.path.join(model_path))
    print('Model loaded!')
    #load files
    if(args.img):
        test_img = utils.preprocess(source_path)
        test_img = np.expand_dims(test_img, axis=0)
        prediction = model.predict(test_img)
        print('Prediction is: ' + str(prediction))
    else:
        import pandas as pd
        csv = pd.read_csv(source_path, header=None)
        print(csv.columns)
        image_list = csv[0].tolist()
        pred_list = []
        label_list = []
        for img in image_list:
            test_img = utils.preprocess(img)
            test_img = np.expand_dims(test_img, axis=0)
            prediction = model.predict(test_img)
            #binarize
            if prediction > 0.5:
                label_list.append(1)
            else:
                label_list.append(0)
            pred_list.append(prediction[0])
        df = pd.DataFrame({'images':image_list,'predictions':pred_list, 'predicted labels':label_list})
        if (output_csv != None):
            df.to_csv(output_csv)
            print(output_csv + ' created!')
        else:
            raise argparse.ArgumentError('-out', 'please specify output csv name')

def main():
    parser=argparse.ArgumentParser(description="Predict whether or not there is a bud in a given image or barch of images (csv)")
    parser.add_argument('-img', action='store_true',dest="img", help='flag for single image prediction', required=False)
    parser.add_argument("-src",help="absolute path of the image or csv" ,dest="source", type=str, required=True)
    parser.add_argument("-model",help="absolute path to h5 file" ,dest="model", type=str, required=True)
    parser.add_argument('-out',help="absolute path of output csv file with file extension (.csv)" ,dest="output", type=str, required=False)
    parser.set_defaults(func=run)
    
    args=parser.parse_args()
    if (not os.path.exists(args.source)):
        parser.error('Invalid path to source file')
    if (not os.path.exists(args.model)):
        parser.error('Invalid path to h5 file')
    args.func(args)


if __name__=="__main__":
	main()
