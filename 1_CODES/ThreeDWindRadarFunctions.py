import os, glob, re, functools, operator
import pandas as pd
from OtherUsefulFunctions import correct_hour, getNextDate
import datetime as dt
import numpy as np

def readASWradar(chosenDate, path_asw, optionFlag):
    """
    Read .asw files, taking into account measurements that are flags
    If there is no measurements, value is NaN. If measurement is flagged, all velocities are set to 0
    The last 10 minuts of the day will be missing
    Args:
        chosenDate : string of day chosen, as for example '2020619' or '2020720'
        path_asw : directory containing radar final product. One folder per day, containing one file per 10 minuts
    Returns:
        df_asw : dataframe containing data, with columns 'flag','Altitude [m, ASL]','East [m/s]','North [m/s]',
        'Vertical [m/s]','dd [°]','ff [m/s]','Min SNR [dB]','Time_written','Time_start', 'Time_end'
    """
        
    filepaths_asw = sorted(glob.glob(path_asw + '/*'+chosenDate+'*/*'))

    done = False
    # Looping through lines of the file
    for k in range(0, len(filepaths_asw)): 
        data=[]
        with open(filepaths_asw[k], 'rb') as filehandle:
            filecontent = filehandle.readlines()
            for index, line in enumerate(filecontent):

                if 'used raw data :' in line.decode("ISO-8859-1"):
                    data.append(line.decode("ISO-8859-1").replace('\r\n', '').replace('->',''))
                    nameDoc = re.search("([0-9]{7})", filepaths_asw[k]).group(1)
                    try:
                        nameHour = re.search("([0-9]{3}.asw)", filepaths_asw[k]).group(1)[:-4]
                    except AttributeError:
                        nameHour = re.search("([a-zA-Z][0-9]{2}.asw)", filepaths_asw[k]).group(1)[:-4]
                    nameHour = correct_hour(nameHour).zfill(4)

                    startOff = dt.datetime.strptime(nameDoc+str('_')+nameHour[0:2]+':'+nameHour[2:4]+':00', '%Y%m%d_%H:%M:%S')         
                    xx=re.search("([0-9]{2}:[0-9]{2}:[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", data[0]).group(1).split(' ')[0]
                    startDoc = dt.datetime.strptime(nameDoc+str('_')+xx, '%Y%m%d_%H:%M:%S')
                    xx =re.search("([0-9]{2}:[0-9]{2}:[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})", data[0]).group(1).split(' ')[2]
                    endDoc = dt.datetime.strptime(nameDoc +str('_') + xx, '%Y%m%d_%H:%M:%S')

                    if k<20 and startDoc.hour >= 22 and startDoc.hour < 24 :
                        startDoc = startDoc - dt.timedelta(days=1)
                    if k<20 and endDoc.hour >= 22 and endDoc.hour < 24 :
                        endDoc = endDoc - dt.timedelta(days=1)

                if 'Altitude' in line.decode("ISO-8859-1"): 
                    done=True       
                if '* : data invalidate by user or quality control' in line.decode("ISO-8859-1"):
                    done=False
                if done:
                    dataTemp = re.sub('\s+',' ',line.decode("ISO-8859-1").replace('\r\n','').replace('//////','nan')).split(' ')  

                    if dataTemp[0]=='*' and (optionFlag==True):
                        dataTemp[2:5]= [0,0,0]
                    else:
                        try:
                            if dataTemp[2]=='nan':
                                dataTemp[2:5]= [np.float64(np.nan),np.float64(np.nan),np.float64(np.nan)]
                        except IndexError:
                            pass
                    dataTemp.append(startOff)
                    dataTemp.append(startDoc)
                    dataTemp.append(endDoc)
                    data.append(dataTemp)
        columns = ['flag','Altitude [m, ASL]','East [m/s]','North [m/s]','Vertical [m/s]','dd [°]','ff [m/s]',
                   'Min SNR [dB]','Time_written','Time_start', 'Time_end']

        if k==0:
            df_asw = pd.DataFrame(data[3:-1], columns = columns)             
        else:
            df_asw = df_asw.append(pd.DataFrame(data[3:-1], columns = columns))

    return df_asw

def concatASWradar(chosenDate, path_asw, optionFlag):
    """
    Allows to concatenate data from the chosen day and the following day, 
    so you can obtain the last 10 minuts measurements of the chosen day
    Args:
        chosenDate : string of day chosen, as for example '2020619' or '2020720'
        path_asw : directory containing radar final product. One folder per day, containing one file per 10 minuts
    Returns:
        df_asw : dataframe containing data, with columns 'flag','Altitude [m, ASL]','East [m/s]','North [m/s]',
        'Vertical [m/s]','dd [°]','ff [m/s]','Min SNR [dB]','Time_written','Time_start', 'Time_end'
    """
    df_asw = readASWradar(chosenDate,path_asw,optionFlag)
    try:
        nextDate = getNextDate(chosenDate)
        df_Nasw = readASWradar(nextDate,path_asw,optionFlag) 
        dfRetour = df_asw.append(df_Nasw, ignore_index=True)        
    except UnboundLocalError:
        dfRetour = df_asw
    return dfRetour, optionFlag

def ASW2Barb(df_asw,optionFlag):
    """
    Performs operations needed to plot the wind barbs
    Args:
        df_asw : dataframe containing 3D wind measurements (u,v,w,direction,altitude,etc)
    Returns:
        df_barb : dataframe ready to use to plot wind barbs. 
        The speed is added, the flagged measurements have a speed of 100 so they can be clearly visible
    """
    df_barb = df_asw.loc[:, ['Altitude [m, ASL]','East [m/s]','North [m/s]','Vertical [m/s]', 'Time_written']] 
    df_barb.loc[:, df_barb.columns != 'Time_written'] = df_barb.loc[:, df_barb.columns != 'Time_written'].apply(pd.to_numeric, errors='coerce')
    df_barb['Speed'] = np.sqrt(np.square(df_barb['East [m/s]'])+np.square(df_barb['North [m/s]']))
    if optionFlag==True:
        df_barb.loc[df_barb.Speed == 0, 'Speed'] = 100    
    return df_barb

def uvw2radial(df_asw, beam_radar, elevationAngle, mode):
    
    """
    Transforms u,v and w into radial velocity and frequency depending on the beam azimuth (projection)
    For the vertical beam, it is not necessary
    Args:
        df_asw : dataframe containing 3D wind measurements (u,v,w,direction,altitude,etc)
        beam_number : radar beam identification number 
        beam_radar : azimuths, see Variables.py
        elevationAngle : supposed to be 75 degrees, written in files 7
        mode : 'high' or 'low', has to be concordant with the variable path_asw in function concatASWradar 
               (one path for each mode)
    Returns:
        df_radial : dataframe with added radial velocities and frequency
    """
    df_radial=df_asw.copy()
    theta = np.radians(elevationAngle)
    r = [[1,6] if mode == 'high' else [6,11]]
    
    for beam_number in range(r[0][0],r[0][1]):
        phi = np.radians(beam_radar[str(beam_number)])
        if not (beam_number == 1 or beam_number ==6):
            df_radial[str(beam_number)+'speedR'] = df_radial['East [m/s]'].astype(float)*np.cos(theta)*np.sin(phi) + df_radial['North [m/s]'].astype(float)*np.cos(theta)*np.cos(phi) + df_radial['Vertical [m/s]'].astype(float)*np.sin(theta)     
        else:
            df_radial[str(beam_number)+'speedR'] = df_radial['Vertical [m/s]'].astype(float)
        df_radial[str(beam_number)+'frequencyR'] = (-2*1290/299.79)*df_radial[str(beam_number)+'speedR']  
        
    return df_radial


