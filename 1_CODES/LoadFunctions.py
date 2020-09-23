import os, glob, re, functools, operator, subprocess
from OtherUsefulFunctions import *
import datetime as dt
import matlab.engine
import numpy as np
import matlab.engine
    
def loadRadar(chosenDate, path_dat,verbose=True):
    """
    Loads radar directory content in the form of lists
    Args:
        chosenDate : string of day chosen, as for example '2020619' or 2020720
        path_dat : directory containing radar measurements
    Returns:
        folder_radar : list of subfolders for the day chosen (corresponding to each hour of the day)
        filepaths_radar : list of full paths to the files contained in each of the subfolders
        filenames_radar : list of names only of files contained in each of the subfolders
        date_radar : list of dates for the day chosen, as encoded in the name of the files
        corresponding to beginning of each measurement cycle
        date_radardt : same as date_radar but transformed to datetimes
    """
    
    if not os.path.exists(str(path_dat)+chosenDate+'.DAT/'):
        if verbose is True: print('Directory specified for radar measurements not found for date ' + chosenDate)
        radarNoFile = True
    else:
        radarNoFile = False
            
    if os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"Files.pickle"):
        with open(os.getcwd()+'/Lists/'+chosenDate+"Files.pickle", "rb") as f:
              folder_radar, filepaths_radar,filenames_radar, date_radar, date_radardt = pickle.load(f)
                
    else:
        #Adding last two hours of previous day (if existant)
        previousDate = getPreviousDate(chosenDate)
        try:
            folder_radar=[sorted(glob.glob(str(path_dat)+previousDate+'.DAT/'))[0],
                          sorted(glob.glob(str(path_dat)+chosenDate+'.DAT/'))[0]]
        except IndexError:
            folder_radar=sorted(glob.glob(str(path_dat)+chosenDate+'.DAT/'))
        
        filepaths_radarTemp = [sorted(glob.glob(k+'*.dat')) for k in folder_radar]
        filepaths_radarTemp= functools.reduce(operator.iconcat, filepaths_radarTemp, [])
        filepaths_radar=[]
        eng = matlab.engine.start_matlab()
        eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
        
        for k in filepaths_radarTemp:
            out = eng.loadCheck(k)  
            if (previousDate[2:]+'m' in k) and (out==1):
                filepaths_radar.append(k)
            elif ((previousDate[2:]+'n') in k) and (out==1) :
                filepaths_radar.append(k)
            elif (chosenDate in k) and (out==1):
                filepaths_radar.append(k)
            elif out==0:
                print(k, ' missing beams (less than 10 measurements)')
 
        filenames_radar =[os.path.basename(k) for k in filepaths_radar]
        date_radar = [correct_hour(k[:-4]) for k in filenames_radar]
        date_radardt = [dt.datetime.strptime('20'+k[:5]+k[5:-2].zfill(2)+k[-2:], '%Y%m%d%H%M') for k in date_radar]
        
        with open(os.getcwd()+'/Lists/'+chosenDate+"Files.pickle", "wb") as f:
            pickle.dump([folder_radar, filepaths_radar,filenames_radar, date_radar, date_radardt], f)

    return folder_radar, filepaths_radar,filenames_radar, date_radar, date_radardt, radarNoFile
    
def loadLidar(chosenDate, path_lidar,optionZip=True, verbose=True):
    """
    Loads lidar directory content in the form of lists
    Performs extraction of the compressed files if it hasn't been done yet
    Args:
        chosenDate : string of day chosen, as for example '2020619' or 2020720
        path_lidar : directory containing lidar measurements
    Returns:
        folder_lidar : list of subfolders for the day chosen (corresponding to each hour of the day)
        filepaths_lidar: list of full paths to the files contained in each of the subfolders
        filenames_lidar : list of names only of files contained in each of the subfolders
        date_lidar : list of dates for the day chosen, as encoded in the name of the files, 
        corresponding to beginning of each measurement cycle
        folder_lidarFIXED, filepaths_lidarFIXED, filenames_lidarFIXED, date_lidarFIXED : 
        same as abovementionned but for the   first 6 minuts of each hour during which the lidar measures in a fixed direction 
        (configuration of July 2020)
    """    
    if not os.path.exists(path_lidar):
        if verbose is True: print('Directory specified for lidar measurements not found for date ' + chosenDate)
        lidarNoFile = True 
        
    else:
        lidarNoFile = False
        #Adding last two hours of previous day (if existant)
        previousDate = getPreviousDate(chosenDate)        
        folder_lidar = [sorted(glob.glob(str(path_lidar)+previousDate+'/wind_and_aerosols_data/2'+str(x)+'-00/')) for x in [2,3]]
        folder_lidar.append(sorted(glob.glob(str(path_lidar)+chosenDate+'/wind_and_aerosols_data/*/'), 
                              key=lambda x:int(re.search("(/\d{1,2}-)", x).group(1)[1:-1])))
        folder_lidar=functools.reduce(operator.iconcat, folder_lidar, [])

        #Unzip files if needed
        filepaths_gz = [sorted(glob.glob(str(k)+'WLS*.gz')) for k in folder_lidar] 
        filepaths_gz = functools.reduce(operator.iconcat, filepaths_gz, [])
        if optionZip is True:
            if verbose is True: print('Unzipping lidar files, might take a bit of time') if filepaths_gz else 0
            for k in filepaths_gz:
                try:
                    subprocess.check_call(["gunzip", k])
                except subprocess.CalledProcessError as e:
                    #print('Except Error, check ', k)
                    pass

        filepaths_lidar = [sorted(glob.glob(str(k)+'WLS*dbs*.nc')) for k in folder_lidar]    
        filepaths_lidar = functools.reduce(operator.iconcat, filepaths_lidar, [])   
        filenames_lidar=[os.path.basename(k) for k in filepaths_lidar]
        date_lidar=[re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", k).group(1) 
                    for k in filenames_lidar] 
        date_lidardt = [dt.datetime.strptime(k, '%Y-%m-%d_%H-%M-%S') for k in date_lidar]
        
        filepaths_lidarFIXED = [sorted(glob.glob(str(k)+'WLS*fixed*.nc')) for k in folder_lidar]
        filepaths_lidarFIXED = functools.reduce(operator.iconcat, filepaths_lidarFIXED, [])
        filenames_lidarFIXED=[os.path.basename(k) for k in filepaths_lidarFIXED]
        date_lidarFIXED=[re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", k).group(1) 
                         for k in filenames_lidarFIXED]        
           

        return folder_lidar, filepaths_lidar, filenames_lidar, date_lidar,date_lidardt, filepaths_lidarFIXED, filenames_lidarFIXED, date_lidarFIXED, lidarNoFile
    
    
def loadDataPlot(start, end, chosenDate,beam_number, gate, radarStart, radarEnd, JoinedData, option):
    """
    Allows to retrieve the data needed to plot within the chosen interval
    Args:
        start, end : start and end hours, as for example '0001' and '0202'
        chosenDate : string of day chosen, as for example '2020619' or '2020720'
        beam_number : radar beam identification number 
        gate : altitude identification index (0 to 54)
        radarStart : dictionary of start acquisition time for each beam
        radarEnd : dictionary of end acquisition time for each beam
        JoinedData : dictionary of lidar radial speed for each spectrogram (= each radar file, all beams, all gates)
        option : option = 2 will plot a spectrogram with one column out of 2. 
                 Up to 10 it does not change the visualization of the wind  but allows a much faster plotting

    Returns:
        start_str, end_str : chosen start and end as strings
        start_date, end_date : chosen start and end as datetimes
        timeserieR, timeserieI : real and imaginary parts of the time serie
        spectrogram : spectrogram, which can be reduces if you set option to a number greater than 1
        fspec : frequency domain of the spectrogram (600)
        tspec : temporal domain of the spectrogam (544)
        spec : average spectra on acquisition time, after post processing. The ground clutter for example is not visible 
        index : of radar files that are within the time interval chosen
    """
    paddedDate = padChosenDate(chosenDate)
    start_str = str(paddedDate + '-'+start)
    end_str = str(paddedDate + '-'+ end)
    
    # Select files
    start_date = dt.datetime.strptime(start_str, '%Y%m%d-%H%M')
    end_date = dt.datetime.strptime(end_str, '%Y%m%d-%H%M')
    
    mask = (pd.Series(radarStart[beam_number]) >= start_date) & (pd.Series(radarEnd[beam_number]) <= end_date)
    index = mask[mask==True].index.values
    chosenfilenames = JoinedData['filename'][index[0]:index[-1]+1]
    

    #Load
    timeserieR = [];timeserieI=[];spectrogram=[];fspec=[];tspec=[];spec=[];fs=[];fMin=[];fMax=[]    
    eng = matlab.engine.start_matlab()
    eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
    for k in chosenfilenames: 
        out = eng.loadSpec(k, beam_number,gate+1,nargout=7)    
        timeserieR.append(np.array(out[0]).reshape(-1))
        timeserieI.append(np.array(out[1]).reshape(-1))    
        spectrogram.append(np.array(out[2])[:,np.arange(0,544,option)])
        fspec.append(np.array(out[3]).reshape(-1))
        fMin.append(np.min(np.array(out[3]).reshape(-1)))
        fMax.append(np.max(np.array(out[3]).reshape(-1)))
        tspec.append(np.array(out[4]).reshape(-1)[np.arange(0,544,option)])
        spec.append(np.array(out[5]).reshape(-1))
        fs.append(np.array(out[6])*1)

    return start_str, end_str,start_date, end_date, timeserieR,timeserieI, spectrogram, fspec, tspec, spec, index, fs, fMin, fMax
    
    

        
        