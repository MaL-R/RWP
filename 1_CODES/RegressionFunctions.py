import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import glob, os, re, pickle, random, argparse
import pandas as pd
import numpy as np
from Paths import *
from OtherUsefulFunctions import *
from Variables import *
import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,Dropout, 
                                     Flatten, Input, MaxPool2D, Permute, Concatenate,AveragePooling2D)
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
    parser.add_argument('--load_size', type=int, default=50)
    parser.add_argument('-lists_set', type=str, default=os.getcwd()+'/Manual_classificationLAST/reg_lists.pickle')
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
    Test days for classification : 2020628, 2020714, 2020710 and 2020711
    Args:
        args : output of function parse_args()
        reCreate : If lists already created, saved in foler Manual_classificationLAST 
                   Reloaded automatically, unless reCreate is set to True
                   The results which can be found in the report are based on those original lists
    Returns:
        train_list, test_list, val_list :  made up of one list for each sample
        Validation list is a random selection of the train list
        Organised as follows [ [gate, hour, beam, filename, class, lidar frequency], 
                               [gate, hour, beam, filename, class, lidar frequency],
                               ...]  
    """
    if os.path.exists(args.lists_set) and (reCreate is False):
        if args.verbose:
            print("Loading data list from pickle files")
            
            with open(os.getcwd()+'/Manual_classificationLAST/reg_labels.pickle', "rb") as f:
                labels_train, labels_test,labels_val = pickle.load(f)
            
            with open(args.lists_set, "rb") as f:
                train_list,test_list,val_list = pickle.load(f)
    else:
        if args.verbose:
            print("No train & test lists pickles")
            print("Loading and generating pickles")
            
        filenames = sorted(glob.glob(pathManualClass +'*.csv'))
        filenames_test = [k for k in filenames if '2020628' in k]+[k for k in filenames if '2020714' in k]+[k for k in filenames if '2020710' in k]+[k for k in filenames if '2020711' in k]

        filenames_train = [k for k in filenames if k not in filenames_test]
        
        labels_traintemp, train_listtemp = load_labels_filenames(filenames_train)
        labels_test, test_list = load_labels_filenames(filenames_test)
        
        # Shuffle and split
        random.shuffle(test_list)
        temp = []
        for indice, k in enumerate(train_listtemp):
            temp.append([k,labels_traintemp[indice]])
        random.shuffle(temp)
        listTemp=[];labelsTemp=[];
        for indice,k in enumerate(temp):
            listTemp.append(k[0])
            labelsTemp.append(k[1])
        labelsTemp=np.array(labelsTemp) 
        train_list = listTemp[300:]; labels_train=labelsTemp[300:]
        val_list = listTemp[0:300]; labels_val=labelsTemp[0:300]
                  
        # Save files as pickle
        with open(args.lists_set, "wb") as f:
            pickle.dump([train_list,test_list, val_list], f)
            
        with open(os.getcwd()+'/Manual_classificationLAST/reg_labels.pickle', "wb") as f:
            pickle.dump([labels_train, labels_test,labels_val], f)  
    
    labels_train1 = labels_train[:2650]
    labels_test = labels_test[:350]
    train_list1 = train_list[:2650]
    test_list = test_list[:350]
                                                
    if args.verbose:
        print('#####################################################\nDataset Statistics')
        print('Train, test and validation lists : ', len(train_list1), '&', len(val_list), '&', len(test_list))
        print('Train size [%] : ', round(len(train_list1)/(len(train_list1)+len(test_list)+len(val_list))*100,1))
        print('Validation size [%] : ', round(len(val_list)/(len(train_list1)+len(test_list)+len(val_list))*100,1))
        print('Test size [%] : ', round(len(test_list)/(len(train_list1)+len(test_list)+len(val_list))*100,1))

        for k in [1,3]:
            total = len(labels_train1[labels_train1==k])+len(labels_test[labels_test==k])+len(labels_val[labels_val==k])
            print('Class ', str(k), ' : ', total,
                  ' Train ', str(round(np.count_nonzero(labels_train1==k)/total*100,1)).rjust(4,' '), 
                  ' Val ', str(round(np.count_nonzero(labels_val==k)/total*100,1)).rjust(4,' '), 
                  ' Test ', str(round(np.count_nonzero(labels_test==k)/total*100,1)).rjust(4,' '), 
                 )
        print('#####################################################')
           
    return train_list, test_list, val_list

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
            df_labels.where(df_labels!=2, inplace=True)
            df_labels.where(df_labels!=0, inplace=True)

            labelsTemp = df_labels.stack().values
            labels = np.append(labels, labelsTemp)
                     
        except FileNotFoundError:
            print('No JoinedData list for day : ', chosenDate)
            pass
        
        toLoadTemp=[];
        for indice, k in enumerate(list(df_labels.stack().index)):  
            # gate, hour, beam, filename, class, lidar frequency
            try:
                beam_number = int(re.search("Beam([1-5]{1})", file).group(1))
                position = [i for i,s in enumerate(filenameList) if chosenDate[2:]+correct_name(k[1]) in s] 
                if not np.isnan(JoinedData[beam_number][position[0]][k[0]]):                   
                    frequency = v2fRadar(JoinedData[beam_number][position[0]][k[0]])                
                    toLoadTemp.append(list(k))
                    toLoadTemp[indice].append(beam_number)
                    toLoadTemp[indice].append(filenameList[position[0]])
                    toLoadTemp[indice].append(labelsTemp[indice])
                    toLoadTemp[indice].append(frequency)

                    toLoad.append(toLoadTemp[indice])
                    labelsBis = np.append(labelsBis, labelsTemp[indice])

                    toLoad.append([toLoadTemp[indice][0],toLoadTemp[indice][1],toLoadTemp[indice][2],
                                 toLoadTemp[indice][3],toLoadTemp[indice][4]*-1,toLoadTemp[indice][5]*-1])  
                    labelsBis = np.append(labelsBis, labelsTemp[indice])

            except IndexError:
                pass
                
    return labelsBis, toLoad

###################################################################################################################
# TFRECORDS

def create_tfrecords_reg(either_list, args, name, reCreate=False):
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
        list_fs : additional features values for each sample. 
                  It is faster than having to loop again through all tfrecords if you need to get the minimum and maximum 
                  on the train set for example, or if you need them to plot test results.
    """
    list_tfrecord=[];list_fs=[]
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
    
    directory = pathBase + '3_Regression/reg_tfrecords/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #either_list = either_list[:int(len(either_list)/args.batch_size)*args.batch_size]
    split_list = list(np.arange(args.load_size,len(either_list),args.load_size))
    either_list_split = [either_list[i : j] for i, j in zip([0] + split_list, split_list + [None])] 
    
    count = 0
    for k in range(len(either_list_split)+1): 
        if os.path.isfile(directory + name + str(k) +'.tfrecord'):
            count = count+1
            
    if (count == len(either_list_split)) and (reCreate is False):
        if args.verbose: print('The ' + str(count) + ' .tfrecord you are about to produce already exist, probably for the same list.\nRe-run and change name if you do not want to overwrite them. If you do, set reCreate to True')
        with open(directory+name+'_list.pickle', "rb") as f:
            list_tfrecord,list_fs = pickle.load(f)
            
    elif (reCreate is True) or (count != len(either_list_split)):
        if args.verbose and (reCreate is True): print(str(count) + ' .tfrecord exist with same name.\nYou are overwritting them, reCreate is set to True')
        if args.verbose and (count != len(either_list_split)) and (count!=0) and (reCreate is not True): print(str(count) + ' .tfrecord exist with same name.\nYou are about to produce ' + str(len(either_list_split)) + ' (different number of files, new list probably)')
        if args.verbose and (count==0): print('No existing .tfrecord. You are about to produce ' + str(len(either_list_split)))
        
            
        for indice1, k1 in enumerate(either_list_split):
            spectrogram = np.zeros((len(k1),INPUT_HEIGHT, INPUT_WIDTH));
            labels = np.array([])
            fs = np.array([]) 
            fmax = np.array([])
            output1= np.zeros((len(k1),10));output2= np.zeros((len(k1),20));
            
            for indice2, k2 in enumerate(k1):
                out = eng.loadSpecReg(k2[3].replace('/scratch/maelle/radar/', path_dat),k2[2],k2[0]+1, nargout=4)
                
                if k2[4] >= 0:
                    spec =  np.array(out[0])
                if k2[4] < 0:
                    spec = np.flip(np.array(out[0]),axis=0)
                    k2[4]=k2[4]*-1
                
                spectrogram[indice2,:,:] = spec
                fs = np.append(fs, out[1])
                labels = np.append(labels, k2[5]/out[2])
                fmax = np.append(fmax, out[2])
                
                index = np.digitize(k2[5],np.arange(-40,51,10))
                result = np.zeros(10)
                result[index]=1
                output1[indice2,:] = result

                index = np.digitize(k2[5],np.arange(-45,51,5))
                result = np.zeros(20)
                result[index]=1
                output2[indice2,:] = result
                
                if args.verbose: drawProgressBar((indice2+1)/args.load_size, barLen=25)

            write_tfrecords_file_reg(directory+name+str(indice1)+'.tfrecord', 
                                 spectrogram.reshape(len(k1),INPUT_HEIGHT, INPUT_WIDTH,1).astype(np.float32), 
                                 labels.astype(np.float32),
                                 fmax.astype(np.float32), 
                                 output1.astype(np.float32), output2.astype(np.float32)) 
            
            list_tfrecord.append(directory+name+str(indice1)+'.tfrecord')
            list_fs.append(fs)
            
            with open(directory+name+'_list.pickle', "wb") as f:
                pickle.dump([list_tfrecord,list_fs], f)   
                
            if args.verbose: print('Created : ', name+str(indice1)+'.tfrecord (' + str(len(either_list_split)-(indice1 +1)) + ' more to go)')
      
    if '/scratch/maelle/' in list_tfrecord[0]:
        list_tfrecordNew=[]
        for k in list_tfrecord:
            list_tfrecordNew.append(k.replace('/scratch/maelle/class_tfrecords/', pathBase + '3_Regression/reg_tfrecords'))
    else:
        list_tfrecordNew=list_tfrecord
                
    return list_tfrecordNew,list_fs

def write_tfrecords_file_reg(out_path, images, labels, fweights, Hot1, Hot2):
    """
    Function used in create_tfrecords_reg which writes the tfrecord
    Requires following helper functions _int64_feature and _bytes_feature
    Args:
        images : spectrograms computed from the time serie with a discrete Gabor transform (Matlab function)
        labels : lidar Doppler frequency shift, target
        fweight : maximum of the spectrogram frequency range
        Hot1, Hot2 : those were trials to obtain a prediction of the Doppler frequency shift with a softmax activation. 
        The lidar Doppler frequency shift falls in a bin (10 in total for Hot1 and 20 for Hot2) as is hot encoded, 
        so can be treated as a class. That brings the number of classes to 10 and 20 respectively and would require 
        a higher number of samples. 
    """    
    assert len(images) == len(labels)
    with tf.io.TFRecordWriter(out_path) as writer:  # could use writer_options parameter to enable compression
        for i in range(len(labels)):
            img_bytes = images[i].tostring()  
            labels_temp = labels[i].tostring() 
            fweight_temp= fweights[i].tostring()
            Hot1_temp= Hot1[i].tostring()
            Hot2_temp= Hot2[i].tostring()
            data = {'image': _bytes_feature(img_bytes), 
                    'label': _bytes_feature(labels_temp),
                    'fweight':_bytes_feature(fweight_temp),
                    'Hot1':_bytes_feature(Hot1_temp),
                    'Hot2':_bytes_feature(Hot2_temp)}
            feature = tf.train.Features(feature=data)  
            example = tf.train.Example(features=feature)  
            serialized = example.SerializeToString()  
            writer.write(serialized)  
            
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value): # feature is a scalar
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
def _float_array_feature(value): # feature is a vector
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def normalize_with_moments(x,axes=[0,1],epsilon=1e-8):
    mean,variance=tf.nn.moments(x,axes=axes)
    x_normed = (x-mean)/tf.sqrt(variance+epsilon)
    return x_normed

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
    dataset = dataset.map(parse_example_reg).shuffle(buffer_size=1) # list already shuffled when created
    dataset = dataset.repeat()
    dataset = dataset.batch(args.batch_size)
    iterator = dataset.make_one_shot_iterator()
    dataset= iterator.get_next()
    return dataset

def parse_example_reg(serialized, shape=(INPUT_HEIGHT, INPUT_WIDTH, 1)):
    """
    Parses the tensors and additional features of the tfrecords files
    Used in the input pipeline
    """   
    features = {'image': tf.io.FixedLenFeature([], tf.string), 
                'label': tf.io.FixedLenFeature((), tf.string),
                'fweight': tf.io.FixedLenFeature((), tf.string)}
    # Parse the serialized data so we get a dict with our data.
    parsed_example = tf.io.parse_single_example(serialized=serialized, features=features)
    image_raw = parsed_example['image']  
    image = tf.decode_raw(image_raw, tf.float32) 
    image = tf.reshape(image, shape=shape)
    #image = normalize_with_moments(image)
    label_raw = parsed_example['label']
    label = tf.decode_raw(label_raw, tf.float32)
    label= tf.reshape(label, shape=(1,))    
    fweight_raw = parsed_example['fweight']
    fweight = tf.decode_raw(fweight_raw, tf.float32)
    fweight = tf.reshape(fweight, shape=(1,))
    fweight_norm = normalize_fixed(fweight, current_min=51.75983436853002, current_max=178.57142857142858,
                                  normed_min=0, normed_max=1)
    
    return (image, fweight_norm, fweight), label

def normalize_fixed(x,current_min,current_max, normed_min,normed_max):
    """
    Min/max normalisation of additional feature maximum of the spectrogram frequency range
    It has the following minimum and maximum on the train set : [51.75983436853002, 178.57142857142858]
    Used in parsing function 
    """  
    x_normed = (x-current_min)/(current_max-current_min)
    x_normed = x_normed*(normed_max-normed_min)+normed_min
    return x_normed


###################################################################################################################
# MODELS

def ConvModelx(): 

    input_shape=(INPUT_WIDTH,INPUT_HEIGHT,1)
       
    y = Input(shape=input_shape, name='Spectrogram', dtype='float32')
    y2 = Input(shape=(1,), name='FreqMaxNorm', dtype='float32')
    fweight = Input(shape=(1,), name='FreqMax', dtype='float32')
            
    x = Conv2D(16, (3, 3), padding="same",activation='relu')(y)
    x = Conv2D(16, (3, 3), padding="same",activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(32, (3, 3), padding="same",activation='relu')(x)
    x = Conv2D(32, (3, 3), padding="same",activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same",activation='relu')(x)
    x = Conv2D(64, (3, 3), padding="same",activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same",activation='relu')(x)
    x = Conv2D(128, (3, 3), padding="same",activation='relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16,activation='relu')(x)
    x = Dense(16,activation='relu')(x)
    x = Concatenate(name='Layer5')([x,y2])
    x = Dense(1,activation='linear')(x)

    return Model(inputs=[y,y2,fweight], outputs=x), fweight

###################################################################################################################
# ADDITIONAL FUNCTIONS

def getFspec(fss):
    """
    Retrieves frequency bins/axis from the sampling frequency fss
    """  
    fspec1 = np.array([-1+2/600, 1])
    fspec2 = fspec1*fss/2
    fspec3 = np.linspace(fspec2[0], fspec2[1], 600)
    fmax = fspec2[1]
    return fmax

def custom_loss_wrapper(fweight):
    """
    Custom loss function allowing the retrieval of the mean absolute error in Hz
    Which can then be converted in Doppler velocity shift (m/s) 
    """  
    def custom_mae(y_true, y_pred):
        return K.mean(fweight*K.abs(y_true-y_pred), axis=-1)
    return custom_mae




