import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import glob, os, re, pickle, random, argparse
import pandas as pd
import numpy as np
from Paths import *
from OtherUsefulFunctions import *
from Variables import *
from LoadFunctions import loadRadar
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,Dropout, 
                                     Flatten, Input, MaxPool2D, Permute, Concatenate,Multiply,AveragePooling2D)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matlab.engine

def parse_args():
    """
    Parses the args given by the user
    """  
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
            
    parser = argparse.ArgumentParser()
    parser.add_argument('-f')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--load_size', type=int, default=100)
    parser.add_argument('-lists_set', type=str, default=os.getcwd()+'/Manual_classificationLAST/class_lists.pickle')
    parser.add_argument('--skip_train', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--skip_test', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--load_weights', type=str2bool, nargs='?',
                    const=True, default=False)
    parser.add_argument('--model_number', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default=path_dat)

    args = parser.parse_args()
    return args

def load_lists(args, reCreate=False):
    """
    Creates lists based on manual classification files
    Test days for classification : 2020711 and 2020712 
    Args:
        args : output of function parse_args()
        reCreate : If lists already created, saved in foler Manual_classificationLAST 
                   Reloaded automatically, unless reCreate is set to True
                   The results which can be found in the report are based on those original lists
    Returns:
        train_list and test_list :  made up of one list for each sample
        Validation list is a random selection of the train list, see code 4_Classification
        Organised as follows [ [gate, hour, beam, filename, class, lidar frequency], 
                               [gate, hour, beam, filename, class, lidar frequency],
                               ...]  
    """
    if os.path.exists(args.lists_set) and (reCreate is False):
        if args.verbose:
            print("Loading data list from pickle files")
            
            with open(os.getcwd()+'/Manual_classificationLAST/class_labels.pickle', "rb") as f:
                labels_train, labels_test = pickle.load(f)
            
            with open(args.lists_set, "rb") as f:
                train_list,test_list = pickle.load(f)
    else:
        if args.verbose:
            print("No train & test lists pickles")
            print("Loading and generating pickles")
            
        filenames = sorted(glob.glob(pathManualClass +'*.csv'))
        filenames_test = [k for k in filenames if '2020711' in k]+[k for k in filenames if '2020712' in k]
        filenames_train = [k for k in filenames if k not in filenames_test]
        labels_train, train_list = load_labels_filenames(filenames_train)
        labels_test, test_list = load_labels_filenames(filenames_test)
        random.shuffle(train_list);random.shuffle(test_list)
                  
        # Save files as pickle
        with open(args.lists_set, "wb") as f:
            pickle.dump([train_list,test_list], f)
            
        with open(os.getcwd()+'/Manual_classificationLAST/class_labels.pickle', "wb") as f:
            pickle.dump([labels_train, labels_test], f)

    labels_train1 = labels_train
    labels_train=np.array(list(labels_train[0:3600])+list(labels_train[4400:6800]))
    labels_validation = labels_train1[3600:4400]
    labels_test=labels_test[0:900]
    
    if args.verbose:

        print('######################################################################################\nDataset Statistics')
        print('Train and test lists done, sizes : ', len(labels_train),'&', len(labels_validation), '&', len(labels_test))
        print('Test size [%] : ', round(len(labels_test)/(len(labels_train)+len(labels_test)+len(labels_validation))*100,1))
        print('Validation size [%] : ', round(len(labels_validation)/(len(labels_train)+len(labels_test)+len(labels_validation))*100,1))
        for k in range(0,4):
            print('Class ', str(k), ':   ', 
                  ' Train ', str(round(np.count_nonzero(labels_train==k),3)).rjust(4,' '),
                  ' Test ', str(round(np.count_nonzero(labels_test==k),3)),' [%] ',
                   str(round(np.count_nonzero(labels_test==k)/(np.count_nonzero(labels_test==k)+np.count_nonzero(labels_train==k)+np.count_nonzero(labels_validation==k))*100,1)).rjust(4,' '),

                  ' Validation ', str(round(np.count_nonzero(labels_validation==k),3)),' [%] ',
                   round(np.count_nonzero(labels_validation==k)/(np.count_nonzero(labels_test==k)+np.count_nonzero(labels_train==k)+np.count_nonzero(labels_validation==k))*100,1))
        print('######################################################################################')
            
    return train_list, test_list

def load_labels_filenames(filenames):
    """
    Used in function load_lists
    Goes through the manual classification files, discards samples from classes 4 (flagged rain based on ground measurements)
    and 5 (no class), as well as the NaNs (no classification at all)    
    """  
    labels = np.array([]);labelsBis=np.array([])
    toLoad = []
    for file in filenames:
        try:
            chosenDate = re.search("/([0-9]{7})_", file).group(1)
        except AttributeError:
            chosenDate = re.search("/([0-9]{8})_", file).group(1) 
        
        try:
            with open(os.getcwd()+'/Lists/'+chosenDate+"JoinedData.pickle", "rb") as f:
                JoinedData = pickle.load(f)
            if path_dat not in JoinedData['filename'][0]:
                JoinedData = replacePath(JoinedData, chosenDate, path_dat) 
            filenameList = JoinedData['filename']
              
            df_labels = pd.read_csv(file,index_col=0)
            df_labels.where(df_labels!=5, inplace=True)
            df_labels.where(df_labels!=4, inplace=True)

            labelsTemp = df_labels.stack().values
            labels = np.append(labels, labelsTemp)
                     
        except FileNotFoundError:
            print('No JoinedData list for day : ', chosenDate)
            pass
        
        toLoadTemp=[];
        for indice, k in enumerate(list(df_labels.stack().index)):  
            # gate, hour, beam, filename, class, lidar frequency
            toLoadTemp.append(list(k))
            beam_number = int(re.search("Beam([1-5]{1})", file).group(1))
            toLoadTemp[indice].append(beam_number)
            position = [i for i,s in enumerate(filenameList) if chosenDate[2:]+correct_name(k[1]) in s]      
            toLoadTemp[indice].append(filenameList[position[0]])
            toLoadTemp[indice].append(labelsTemp[indice])
            try:
                toLoadTemp[indice].append(v2fRadar(JoinedData[beam_number][position[0]][k[0]]))
            except IndexError:
                toLoadTemp[indice].append(np.nan)               
            toLoad.append(toLoadTemp[indice])
            labelsBis = np.append(labelsBis, labelsTemp[indice])

            if labelsTemp[indice] == 3.0:
                try:
                    toLoadTemp[indice].append(v2fRadar(JoinedData[beam_number][position[0]][k[0]]))
                except IndexError:
                    toLoadTemp[indice].append(np.nan)
                toLoad.append([toLoadTemp[indice][0],toLoadTemp[indice][1],toLoadTemp[indice][2],
                             toLoadTemp[indice][3],toLoadTemp[indice][4]*-1,toLoadTemp[indice][5]]) 
                labelsBis = np.append(labelsBis, labelsTemp[indice])              
    return labelsBis, toLoad

###################################################################################################################
# TFRECORDS

def create_tfrecords_classADD(either_list, args, name, reCreate=False):
    """
    Loads spectrograms and labels given their filenames
    Args:
        either_list: output of function load_lists so either train, test or validation list
        args : output of function parse_args()
        name : name you want to give to the tfrecord files. 
               If you chose "train" for examples, an increment will be added to the name for each file
               Example : train0, train1, train2, etc.
        reCreate : if there are existing files with the same name and a total number equal to your request, 
                   they will not be recreated. Only the necessary output variables will be loaded from pickle files.
    Returns:
        list_tfrecord : list of tfrecords filenames
        list_noise, list_maxT, list_fs : additional features values for each sample. 
        It is faster than having to loop again through all tfrecords if you need to get the minimum and maximum 
        on the train set for example, or if you need them to plot test results.
    """
    list_tfrecord=[];list_noise=[]; list_maxT=[]; list_fs=[];
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
    
    directory = pathBase + '2_Classification/class_tfrecords/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    split_list = list(np.arange(args.load_size,len(either_list),args.load_size))
    either_list_split = [either_list[i : j] for i, j in zip([0] + split_list, split_list + [None])] 
    
    count = 0
    for k in range(len(either_list_split)+1): 
        if os.path.isfile(directory + name + str(k) +'.tfrecord'):
            count = count+1
            
    if (count == len(either_list_split)) and (reCreate is False):
        if args.verbose: print('The ' + str(count) + ' .tfrecord you are about to produce already exist, probably for the same list.\nRe-run and change name if you do not want to overwrite them. If you do, set reCreate to True')
        with open(directory+name+'_list.pickle', "rb") as f:
            list_tfrecord, list_noise, list_maxT, list_fs = pickle.load(f)
            
    elif (reCreate is True) or (count != len(either_list_split)):
        if args.verbose and (reCreate is True): print(str(count) + ' .tfrecord exist with same name.\nYou are overwritting them, reCreate is set to True')
        if args.verbose and (count != len(either_list_split)) and (count!=0): print(str(count) + ' .tfrecord exist with same name.\nYou are about to produce ' + str(len(either_list_split)) + ' (different number of files, new list probably)')
        if args.verbose and (count==0): print('No existing .tfrecord. You are about to produce ' + str(len(either_list_split)))
            
        for indice1, k1 in enumerate(either_list_split):
            spectrogram = np.zeros((len(k1),INPUT_HEIGHT, INPUT_WIDTH));
            labels = np.array([]); noise=np.array([]); maxT=np.array([]);fs=np.array([])
            
            for indice2, k2 in enumerate(k1):
                out = eng.loadSpecAdditional(k2[3].replace('/scratch/maelle/radar/', path_dat),k2[2],k2[0]+1, nargout=4)
                
                if k2[4] >= 0:
                    spectrogram[indice2,:,:] = np.array(out[0])
                    labels = np.append(labels, int(k2[4]))
                if k2[4] < 0:
                    spectrogram[indice2,:,:] = np.flip(np.array(out[0]),axis=0)
                    labels = np.append(labels, int(k2[4]*-1))    
                    
                noise = np.append(noise, out[1])
                maxT = np.append(maxT, out[2])
                fs = np.append(fs, out[3])
                
                if args.verbose: drawProgressBar((indice2+1)/args.load_size, barLen=25)

            write_tfrecords_fileADD(directory+name+str(indice1)+'.tfrecord', 
                                 spectrogram.reshape(len(k1),INPUT_HEIGHT,INPUT_WIDTH,1).astype(np.float32),labels.astype(int),
                                 noise.astype(np.float32),maxT.astype(np.float32))
            list_tfrecord.append(directory+name+str(indice1)+'.tfrecord')
            list_noise.append(noise)
            list_maxT.append(maxT)
            list_fs.append(fs)
            with open(directory+name+'_list.pickle', "wb") as f:
                pickle.dump([list_tfrecord, list_noise, list_maxT, list_fs], f)                       
            if args.verbose: print('Created : ', name+str(indice1)+'.tfrecord (' + str(len(either_list_split)-(indice1 +1)) + ' more to go)')
    
    if '/scratch/maelle/' in list_tfrecord[0]:
        list_tfrecordNew=[]
        for k in list_tfrecord:
            list_tfrecordNew.append(k.replace('/scratch/maelle/class_tfrecords/', pathBase + '2_Classification/class_tfrecords/'))
    else:
        list_tfrecordNew=list_tfrecord

    return list_tfrecordNew, list_noise, list_maxT, list_fs

def write_tfrecords_fileADD(out_path: str, images: np.ndarray, labels: np.ndarray, noise: np.ndarray, maxT: np.ndarray) -> None:
    """
    Function used in create_tfrecords_classADD which writes the tfrecord
    Requires following helper functions _int64_feature and _bytes_feature
    Args:
        images : spectrograms computed from the time serie with a discrete Gabor transform (Matlab function)
        noise, maxT : additional features which are the noise level and the time serie maximum
        labels : target
    """    
    assert len(images) == len(labels)
    with tf.io.TFRecordWriter(out_path) as writer:  # could use writer_options parameter to enable compression
        for i in range(len(labels)):
            img_bytes = images[i].tostring()  # Convert the image to raw bites
            data = {'image': _bytes_feature(img_bytes), 'label': _int64_feature(labels[i]),
                    'noise':_bytes_feature(noise[i].tostring()),'maxT':_bytes_feature(maxT[i].tostring())}
            feature = tf.train.Features(feature=data)  
            example = tf.train.Example(features=feature)  
            serialized = example.SerializeToString()  
            writer.write(serialized) 

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

###################################################################################################################
# PARSE

def input_pipeline(list_tfrecord, args):
    """
    Pipeline which parses the dataset from tfrecords and prepares its injection to the model
    Args:
        list_tfrecord : list of tfrecords filenames, output of function create_tfrecords_classADD
        args : output of function parse_args()
    Returns:
        dataset : input to the model
    """  
    dataset = tf.data.TFRecordDataset(list_tfrecord)
    dataset = dataset.map(parse_exampleADD).shuffle(buffer_size=1) # list already shuffled when created
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_one_shot_iterator()
    dataset= iterator.get_next()
    return dataset
            
def parse_exampleADD(serialized, shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)):
    """
    Parses the tensors and additional features of the tfrecords files
    Used in the input pipeline
    """    
    features = {'image': tf.io.FixedLenFeature([], tf.string), 
                'label': tf.io.FixedLenFeature([], tf.int64),
                'noise': tf.io.FixedLenFeature([], tf.string),
                'maxT': tf.io.FixedLenFeature([], tf.string)}
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
    label = parsed_example['label']
    image_raw = parsed_example['image']  
    image = tf.decode_raw(image_raw, tf.float32) 
    image = tf.reshape(image, shape=shape)
    noise_raw = parsed_example['noise'] 
    noise = tf.decode_raw(noise_raw, tf.float32)
    noise = normalize_fixed(noise, current_min=0.005767822265625, current_max=24.875, normed_min=0, normed_max=1)
    noise= tf.reshape(noise, shape=(1,))
    maxT_raw = parsed_example['maxT'] 
    maxT = tf.decode_raw(maxT_raw, tf.float32)
    maxT = normalize_fixed(maxT, current_min=3.78125, current_max=3296.0, normed_min=0, normed_max=1)
    maxT= tf.reshape(maxT, shape=(1,))
    return (image,noise,maxT), tf.one_hot(label, depth=NUMBER_OF_CLASSES)

def normalize_fixed(x,current_min,current_max, normed_min,normed_max):
    """
    Min/max normalisation of additional features noise and maximum of time serie
    Those have the following minimum and maximum on the train set : noise [0.005767822265625,24.875] and maxT [3.78125,3296.0]
    Used in parsing function 
    """  
    x_normed = (x-current_min)/(current_max-current_min)
    x_normed = x_normed*(normed_max-normed_min)+normed_min
    return x_normed



###################################################################################################################
# MODELS

def ConvModel3(n_classes):

    y = Input(shape=(INPUT_HEIGHT,INPUT_WIDTH,1), name='Spectrogram', dtype='float32')
    y2 = Input(shape=(1,),name='Noise');
    y3 = Input(shape=(1,),name='TimeSerie');
    
    x = Conv2D(16, (INPUT_HEIGHT, 3), padding="same",activation='relu',name='Layer1')(y)
    x = AveragePooling2D(pool_size=(2, x.shape[2]),name='Layer2')(x)
    x = Flatten(name='Layer3')(x)
    x = Dense(32, activation='relu',name='Layer4')(x)
    x = Concatenate(name='Layer5')([x,y2,y3])
    x = Dense(n_classes, activation='softmax',name='Ouput')(x)

    return Model(inputs=[y,y2,y3], outputs=x)

def ConvModel4(n_classes):

    y = Input(shape=(INPUT_HEIGHT,INPUT_WIDTH,1), name='Spectrogram', dtype='float32')
    y2 = Input(shape=(1,),name='Noise');
    y3 = Input(shape=(1,),name='TimeSerie');
    
    x = BatchNormalization(axis=1,name='Norm.Frequency')(y)
    x = Conv2D(16, (INPUT_HEIGHT, 3), padding="same",activation='relu',name='Layer1')(x)
    x = AveragePooling2D(pool_size=(2, x.shape[2]),name='Layer2')(x)
    x = Flatten(name='Layer3')(x)
    x = Dense(32, activation='relu',name='Layer4')(x)
    x = Concatenate(name='Layer5')([x,y2,y3])
    x = Dense(n_classes, activation='softmax',name='Ouput')(x)

    return Model(inputs=[y,y2,y3], outputs=x)

def ConvModel5(n_classes):

    y = Input(shape=(INPUT_HEIGHT,INPUT_WIDTH,1), name='Spectrogram', dtype='float32')
    y2 = Input(shape=(1,),name='Noise');
    y3 = Input(shape=(1,),name='TimeSerie');
    
    x = BatchNormalization(axis=1,name='Norm.Frequency')(y) 
    x = Conv2D(16, (INPUT_HEIGHT, 3), padding="same",activation='relu',name='Layer1')(x)
    x = MaxPool2D(pool_size=(2, x.shape[2]),name='Layer2')(x)
    x = Flatten(name='Layer3')(x)
    x = Dense(32, activation='relu',name='Layer4')(x)
    x = Concatenate(name='Layer5')([x,y2,y3])
    x = Dense(n_classes, activation='softmax',name='Ouput')(x)

    return Model(inputs=[y,y2,y3], outputs=x)







    
 