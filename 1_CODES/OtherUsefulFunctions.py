import pandas as pd
import datetime as dt
import sys, os
import numpy as np
import pickle
import itertools
from sklearn.linear_model import LinearRegression
import scipy
from sklearn.metrics import mean_absolute_error, mean_squared_error

def correct_hour(name):
    """
    Transforms radar filenames encoding from letters to numbers
    Args:
        name : Radar filenames encoded with letters
    Returns:
        Radar filename encoded with numbers
    """
    hour_str = '0123456789abcdefghijklmnop';
    correspondance = {hour_str[xx]:str(xx) for xx in range(10,24)}
    return ''.join(correspondance.get(char, char) for char in name)

def correct_name(hour):
    """
    Opposite of function correct_hour
    Transforms radar hour encoded in numbers to letters
    """
    hour_str = '0123456789abcdefghijklmnop';
    correspondance = {str(xx):hour_str[xx] for xx in range(10,24)}
    for xx in range(0,10): correspondance['0'+str(xx)]=str(xx)
    return correspondance[hour[0:2]]+hour[2:4]
    

def v2fRadar(radialVelocity):
    """
    Doppler velocity shift to frequency shift with the wind profiler wavelenght
    """
    return (-2*1290/299.79)*radialVelocity

def f2vRadar(frequency):
    """
    Doppler frequency shift to velocity shift with the wind profiler wavelenght
    """
    return frequency/(-2*1290/299.79)

def padChosenDate(chosenDate):
    """
    Single numbers will be padded with zeros. Example : July is encoded as 7 and will be transformed to 07
    Variable chosenDate is a string
    """
    return chosenDate[:4]+chosenDate[4:-2].zfill(2)+chosenDate[-2:]

def titleChosenDate(chosenDate):
    """
    Returns the chosen day in the right format for figures title
    Variable chosenDate is a string
    """
    paddedDate = padChosenDate(chosenDate)
    return paddedDate[0:4]+'.'+paddedDate[4:6]+'.'+paddedDate[6:8]

def getPreviousDate(chosenDate):
    """
    Get the day before the chosen day 
    Variable chosenDate is a string
    """
    previousDate = dt.datetime.strptime(padChosenDate(chosenDate), '%Y%m%d')- dt.timedelta(days=1)
    return dt.datetime.strftime(previousDate, '%Y%-m%d')

def getNextDate(chosenDate):
    """
    Get the day after the chosen day
    Variable chosenDate is a string
    """
    nextDate = dt.datetime.strptime(padChosenDate(chosenDate), '%Y%m%d')+ dt.timedelta(days=1)
    return dt.datetime.strftime(nextDate, '%Y%-m%d')

def toDb(spectrogram):
    """
    Log transformation to decibels
    Applied to spectrograms, if not already applied in Matlab
    """   
    return 20*np.log10(np.abs(spectrogram)+sys.float_info.min)

def replacePath(JoinedData, chosenDate, path_dat, optionSave=False):
    """
    Replaces the paths from the produced lists if needed
    """   
    for indice, k in enumerate(JoinedData['filename']):
        JoinedData['filename'][indice] = k.replace('/scratch/maelle/radar/', os.path.dirname(os.getcwd())+'/radar/')
    if optionSave == True:
        with open(os.getcwd()+'/Lists/'+chosenDate+"JoinedData.pickle", "wb") as f:
            pickle.dump(JoinedData, f)
    return JoinedData

def drawProgressBar(percent,barLen=25):
    """
    Progress bar for tfrecords generation
    """ 
    sys.stdout.write("\r")
    sys.stdout.write("[{:<{}}] {:.0f}%  ".format("=" * int(barLen * percent), barLen, percent*100))
    sys.stdout.flush()

def produceDateList(startDate, endDate):
    """
    Generates a list of days as strings between two chosen dates.
    """ 
    dateList=[]
    delta = endDate - startDate
    for i in range(delta.days+1):
        day = startDate + dt.timedelta(days=i)
        dateList.append(dt.datetime.strftime(day,'%Y%-m%d'))
    return dateList

def filterPrevious(date,date_radardt):
    """
    Filters previous day that has been added to the lists but is not always necessary
    """ 
    paddedDate = padChosenDate(date)
    start_str = str(paddedDate + '-'+'0000')
    end_str = str(paddedDate + '-'+ '2359')
    start_date = dt.datetime.strptime(start_str, '%Y%m%d-%H%M')
    end_date = dt.datetime.strptime(end_str, '%Y%m%d-%H%M')

    mask = (pd.Series(date_radardt) >= start_date) & (pd.Series(date_radardt) <= end_date)
    index = mask[mask==True].index.values
    return index

def reshapeList(x,y):
    """
    Reshapes a list of lists in a numpy array.
    Used in code 3_CoherencyAnalysis
    """ 
    return np.array(list(itertools.chain(*x))), np.array(list(itertools.chain(*y)))

def doRegression(x, y):
    """
    Performs a linear regression
    Used in code 3_CoherencyAnalysis
    """ 
    roundNum =2
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask].reshape(-1, 1);y=y[mask].reshape(-1, 1)
    model = LinearRegression().fit(x,y)
    ypred2 = model.predict(x)
    _, _, _, _, std = scipy.stats.linregress(y.reshape(-1), ypred2.reshape(-1))
    return f'{model.score(x,y):.2f}', f'{model.coef_[0][0]:.2f}', f'{model.intercept_[0]:.2f}', f'{np.sum(mask)}'.rjust(6,' '), f'{mean_absolute_error(x,y):.2f}', f'{np.sqrt(mean_squared_error(x,y)):.2f}', mask, model.coef_[0][0], model.intercept_[0],mean_absolute_error(x,y),np.sqrt(mean_squared_error(x,y)),np.sum(mask),model.score(x,y)

    