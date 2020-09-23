import pandas as pd
import os, fnmatch
from OtherUsefulFunctions import *


def disable_event():
    """
    Used in manual classification GUI
    """
    pass

def interactiveManageFiles(date_radardt, chosenDate, beam_number):
    """
    Manages the .csv files where the manual classification is stored. 
    If existing, loads it. If not, creates ones with right measurement times. 
    Args:
        date_radardt : list of datetime for the day chosen, as encoded in the name of the files
                       corresponding to beginning of each measurement cycle
                       output of function loadRadar (LoadFunctions.py)
        chosenDate : string of day chosen, as for example '2020619' or 2020720
        beam_number : radar beam identification number 
    Returns:
        name : of csv file for the chosenDate
        df_save : pandas dataframe for the chosendate, with gate numbers as rows and measurement times as columns
        date_toute : hours as datetime
        
    """ 
    directory = os.getcwd() + '/Manual_classificationLAST/'
    if not os.path.exists(directory):
        os.makedirs(directory)
      
    result = []
    for root, dirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, str(chosenDate+'_Beam'+str(beam_number)+'.csv')): #'*.csv'
                result.append(os.path.join(root, name))
                
    name = directory + chosenDate+'_Beam'+str(beam_number)+'.csv'  
    date_toute = [dt.datetime.strftime(k, '%H%M') for k in date_radardt 
                  if k>=dt.datetime.strptime(padChosenDate(chosenDate), '%Y%m%d')]
    if not result:
        df_save = pd.DataFrame(index=np.arange(55), columns=date_toute) 
        df_save.to_csv(name, index = True)
    else:
        df_save = pd.read_csv(name,index_col=0)   
    return name, df_save, date_toute 


def interactivateFlagRainEvents(path_rain, chosenDate, df_save, name, optionHourly=False):  
    """
    Flaggs rain events automatically based on ground measurements (class 4), over the entire gate altitudes
    An option allows to flag a larger 1-hour interval around one positive measurement
    Args:
        path_rain : leading the the .dat file containing ground-based rain measurements
        chosenDate : string of day chosen, as for example '2020619' or 2020720
        df_save : output of function interactiveManageFile 
        name : output of function interactiveManageFile 
        optionHourly : +/- 1 hour around rain measures on the groud is flagged
    Returns:
        df_save : pandas dataframe for the chosendate, with gate numbers as rows and measurement times as columns
        The output now contains cells containing number 4
        
    """     
    year = chosenDate[0:4];month=chosenDate[4:-2];day = chosenDate[-2:]
    columns_to_keep = ['STA','JAHR','MO','TG','HH','MM','93']
    df_rain = pd.read_table(path_rain, sep="\s+", usecols=columns_to_keep, 
                            skiprows=8, dtype={1:'str',2:'str', 3:'str', 4:'str', 5:'str'})
    
    data_rain = df_rain[(df_rain['JAHR'] == year) & (df_rain['MO'] == month) & (df_rain['TG'] == day)]
    data_rain = data_rain.reset_index(drop=True)
    data_rain['Date']=[dt.datetime.strptime(data_rain.loc[i]['JAHR']+data_rain.loc[i]['MO'].zfill(2)
                                            +data_rain.loc[i]['TG'].zfill(2)+data_rain.loc[i]['HH'].zfill(2)
                                            +data_rain.loc[i]['MM'].zfill(2),'%Y%m%d%H%M') for i in range(0,data_rain.shape[0])]
    
    list_df = [];min_date=[];max_date=[];
    index = data_rain[['Date']][data_rain['93']!=0]
    list_df = [d for _, d in index.groupby(index.index - np.arange(len(index)))]
    
    if optionHourly is False:
        for k in range(0,len(list_df)):
            min_date.append(list_df[k]['Date'].min()-pd.Timedelta(10,'m'))
            max_date.append(list_df[k]['Date'].max())#+pd.Timedelta(10,'m'))
    else:
        for k in range(0,len(list_df)):
            min_date.append(list_df[k]['Date'].min()-pd.Timedelta(30,'min'))
            max_date.append(list_df[k]['Date'].max()+pd.Timedelta(30,'min'))
        

    dd = padChosenDate(chosenDate)
    for x in range(0,len(min_date)):
        start_str_rain = str(dd + '-' + str(min_date[x]).replace(":", "")[11:15])
        end_str_rain = str(dd + '-'+ str(max_date[x]).replace(":", "")[11:15])
        start_date_rain = dt.datetime.strptime(start_str_rain, '%Y%m%d-%H%M')
        end_date_rain = dt.datetime.strptime(end_str_rain, '%Y%m%d-%H%M')
        columns2fill = [k for k in df_save.columns 
                        if start_date_rain < dt.datetime.strptime(dd + k, '%Y%m%d%H%M') <= end_date_rain]
        df_save.fillna({x:4.0 for x in columns2fill}, inplace=True)

    df_save.to_csv(name, index = True)
    
    return df_save


def interactiveFindLastClassified(df_save):
    """
    Finds last classified gate and time 
    So you can start right from there when you go on  
    """  
    df_bool = df_save.isnull()
    df_bool = df_bool[df_bool.any(axis=1)].idxmax(axis=1)
    line = df_bool.index[0]
    column = df_bool[line]
    return line, column