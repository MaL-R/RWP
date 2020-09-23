import os, glob, re, functools, pickle
import datetime as dt
from collections import defaultdict
import matlab.engine
import xarray as xr
import numpy as np
import netCDF4
import pandas as pd
import warnings
from OtherUsefulFunctions import f2vRadar
warnings.filterwarnings('ignore', category=RuntimeWarning)


def lidarStartEnd(chosenDate, filepaths_lidar, reCreate,verbose=True):
    """
    Creates dictionaries of start and end acquisition times for the lidar measurements (all beams)
    Details : varying, approximately 8.31 8.31 8.31 8.81 5.82 seconds with 5 being end(vertical beam)-end(last oblique beam)
              files give the end of the acquition time (radar gives start)
              end of acquisition time is considered as being beginning of next beam
              end(vertical beam)-end(last oblique beam) = 5.82 seconds
              configuration July 2020
    
    Args:        
        chosenDate : string of day chosen, as for example '2020619' or '2020720'
        filepaths_lidar : list of full paths to the files contained in each of the subfolders
                          from function loadLidar (LoadFunctions.py)
        reCreate : set to True if you wanna create the lists again even if they already exist
    Returns:
        lidarStart : dictionary of start acquisition time for each beam
        lidarEnd : dictionary of end acquisition time for each beam
        indiceMissing : list of missing measurements (only first beams sometimes, no idea why)
    """

    if os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"lidarStart.pickle") and os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"lidarEnd.pickle") and (reCreate is False):
        if verbose is True: print("Loading pickle files : start and end times from the lidar for day " + chosenDate)
        with open(os.getcwd()+'/Lists/'+chosenDate+"lidarStart.pickle", "rb") as f:
            lidarStart = pickle.load(f)
        with open(os.getcwd()+'/Lists/'+chosenDate+"lidarEnd.pickle", "rb") as f:
            lidarEnd = pickle.load(f)
        indiceMissing = 'variables loaded from pickle lists, delete them and run code again if you wanna know :)'
        
    else:
        print("Producing pickle files : start and end times from the lidar for day " + chosenDate)
        lidarStart = defaultdict(list);lidarEnd = defaultdict(list)
        endTime=0;timestampFixed=0;indiceMissing=[]

        for indice,k in enumerate(filepaths_lidar):
            group_name = list(netCDF4.Dataset(k).groups.keys())[1]
            sw = xr.open_dataset(k, group=group_name,decode_times=False)
            for beam_number in range(0,5):
                keepTrack = endTime
                try:
                    endTime = dt.datetime.strptime(np.array(sw.timestamp)[beam_number], "%Y-%m-%dT%H:%M:%S.%fZ")
                    if not lidarStart:            
                        lidarEnd[beam_number].append(endTime)
                        lidarStart[beam_number].append(endTime-dt.timedelta(seconds=8.3))
                    else:
                        lidarStart[beam_number].append(keepTrack)
                        lidarEnd[beam_number].append(endTime)
                except ValueError:
                    endTime = dt.datetime.strptime(np.array(sw.timestamp)[beam_number][:3]+'0-0'
                                                   +np.array(sw.timestamp)[beam_number][3:], "%Y-%m-%dT%H:%M:%SZ")
                    timestampFixed+=1
                    if not lidarStart:            
                        lidarEnd[beam_number].append(endTime)
                        lidarStart[beam_number].append(endTime-dt.timedelta(seconds=8.3))
                    else:
                        lidarStart[beam_number].append(keepTrack)
                        lidarEnd[beam_number].append(endTime)
                except IndexError:
                    lidarStart[beam_number].append(dt.datetime(1930, 1, 12, 0, 0, 0))
                    lidarEnd[beam_number].append(dt.datetime(1930, 1, 12, 0, 0, 0))
                    # Missing data won't be taken into account
                    endTime = keepTrack+dt.timedelta(seconds=6.8) if beam_number == 4 else keepTrack+dt.timedelta(seconds=8)
                    indiceMissing.append([indice, beam_number])
                    # Verification with sw.ray_index that it's always the last ones missing, not some in the middle   

        print('Fixed timestamps ', timestampFixed)
        print('Missing beams ', len(indiceMissing), 'see list indiceMissing')

        with open(os.getcwd()+'/Lists/'+chosenDate+"lidarStart.pickle", "wb") as f:
            pickle.dump(lidarStart, f)
        with open(os.getcwd()+'/Lists/'+chosenDate+"lidarEnd.pickle", "wb") as f:
            pickle.dump(lidarEnd, f)
     
    return lidarStart, lidarEnd, indiceMissing

def radarStartEnd(chosenDate, filepaths_radar, reCreate, verbose=True):
    """
    Creates dictionaries of start and end acquisition times for the radar measurements (all beams)
    Details : varying between 28 to 36 seconds
              headers from raw files indicates end of acquisition time (lidar is start)
              beginning of acquisition time is considered as being end of previous beam
              configuration July 2020
    
    Args:
        chosenDate : string of day chosen, as for example '2020619' or '2020720'
        filepaths_radar : list of full paths to the files contained in each of the subfolders
                          from function loadRadar (LoadFunctions.py)
        reCreate : set to True if you wanna create the lists again even if they already exist
    Returns:
        radarStart : dictionary of start acquisition time for each beam
        radarEnd : dictionary of end acquisition time for each beam
    """   
    keepTrack=[]             
    if os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"radarStart.pickle") and os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"radarEnd.pickle") and (reCreate is False):
        if verbose is True: print("Loading pickle files : start and end times from the radar for day " + chosenDate)
        with open(os.getcwd()+'/Lists/'+chosenDate+"radarStart.pickle", "rb") as f:
            radarStart = pickle.load(f)
        with open(os.getcwd()+'/Lists/'+chosenDate+"radarEnd.pickle", "rb") as f:
            radarEnd = pickle.load(f)  

    else:
        print("Producing pickle files : start and end times from the radar for day " + chosenDate)
        
        eng = matlab.engine.start_matlab()
        eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
        radarStart = defaultdict(list);radarEnd = defaultdict(list);count=0;

        for indice, k in enumerate(filepaths_radar): 
            for beam_number in range(1,11):  
                out = eng.loadtime(k, beam_number,nargout=6)                 
                startTime = dt.datetime(out[0],out[1],out[2],out[3],out[4],out[5])
                radarStart[beam_number].append(startTime)           
                if count>0 and beam_number !=1:
                    radarEnd[beam_number-1].append(startTime)
                if count>0 and beam_number==1:
                    radarEnd[10].append(startTime)
                count+=1
        radarEnd[beam_number].append(startTime+dt.timedelta(seconds=30))   

        with open(os.getcwd()+'/Lists/'+chosenDate+"radarStart.pickle", "wb") as f:
            pickle.dump(radarStart, f)
        with open(os.getcwd()+'/Lists/'+chosenDate+"radarEnd.pickle", "wb") as f:
            pickle.dump(radarEnd, f)

    return radarStart, radarEnd

def matchTimes(startTimeR, endTimeR, listStartL, listEndL, lengthAll):                  
    """
    Matches radar and lidar acquition times, for same beam azimuth
    Varies from 28 to 36 seconds for the radar and from 5 to 8 for the lidar (configuration July 2020)
    Three scenarios : 1. Single lidar acquisition within ~ 30 seconds radar acquisition
                      2. Several lidar acquisitions within ~ 30 seconds, radial speed is averaged
                      3. No overlap between the both instruments, closest acquisition time is selected
    Note : I don't know what happened the day I coded that part, it's not optimal and would deserve a review :)
    Args: from functions lidarStartEnd & radarStartEnd
        startTimeR : radarStart[beam_number][indice]
        endTimeR : radarEnd[beam_number][indice]
        listStartL : lidarStart[radar2lidar[str(beam_number)]]
        listEndL : lidarEnd[radar2lidar[str(beam_number)]]
        lengthAll : length of list that want to be matched (radar here)
    Returns:
        index1 : index of lidar acquisition time(s) selected. It can be a single value (scenarios 1&3) or a vector (scenario 2)
    """  
    index1=np.array([]);
    for indice, k in enumerate(listStartL):
        if k >= startTimeR:
            # Finds the first lidar acquisition beginning after the radar start acquisition time
            # Also checks if the 2 previous and 2 next lidar acquisitions lie within the radar acquisition time
            # Stores chronologically
            if startTimeR<listEndL[indice-2] and listStartL[indice-2].time()!= dt.datetime(1930,3,3,0,0,0).time():
                index1 = np.append(index1, listStartL.index(listStartL[indice-2])) 
            if startTimeR<listEndL[indice-1] and listStartL[indice-1].time()!= dt.datetime(1930,3,3,0,0,0).time():
                index1 = np.append(index1, listStartL.index(listStartL[indice-1]))             
            if endTimeR > k:
                # first lidar acquisition needs to be inside the radar acquisition time
                index1 = np.append(index1, listStartL.index(k))  
            if indice<lengthAll-1:
                if endTimeR  > listStartL[indice+1] and listStartL[indice+1].time()!= dt.datetime(1930,3,3,0,0,0).time():
                    index1 = np.append(index1, listStartL.index(listStartL[indice+1])) 
            if indice<lengthAll-2:
                if endTimeR  > listStartL[indice+2] and listStartL[indice+2].time()!= dt.datetime(1930,3,3,0,0,0).time():
                    index1 = np.append(index1, listStartL.index(listStartL[indice+2])) 
            if index1.size == 0:      
                index1 = np.append(index1, listStartL.index(k)) 
                # Maybe you wanna compare which one is really the closest, sometimes it's index -1 or index +1
                # The differences are usually small
            break
    return index1
                  
def retrieveLidarRadialSpeed(filepaths_radar, filepaths_lidar,radarStart, radarEnd, lidarStart, lidarEnd, radar2lidar, chosenDate, high_mode,low_mode, reCreate, verbose=True):   
    """
    Creates dictionaries of lidar radial speed for each radar files (all beams, all gates). 
    Details : configuration July 2020
              Lidar resolution of 100 meters (distance to ground)
              Radar high mode, resolution 143 to 144 meters (distance to ground)
              Radar low mode, resolution 57 to 58 meters (distance to ground)
              Altitudes are linearly interpolated
    
    Args:
        filepaths_radar : list of full paths to the files contained in each of the subfolders
                          from function loadRadar (LoadFunctions.py)
        radarStart, radarEnd : dictionaries of radar start and end acquisition times for each beam
                               output of function radarStartEnd
        lidarStart, lidarEnd : dictionaries of lidar start and end acquisition times for each beam
                               output of function lidarStartEnd
        chosenDate : string of day chosen, as for example '2020619' or '2020701'
        reCreate : set to True if you wanna create the lists again even if they already exist
    Returns:
        JoinedData : dictionary of lidar radial speed for each spectrogram (= each radar file, all beams, all gates)
    """     
    if os.path.exists(os.getcwd()+'/Lists/'+chosenDate+"JoinedData.pickle") and (reCreate is False):
        if verbose is True: print("Loading pickle files : matched radar and lidar acquisition times for day " + chosenDate)
        with open(os.getcwd()+'/Lists/'+chosenDate+"JoinedData.pickle", "rb") as f:
            JoinedData = pickle.load(f)
    else:         
        eng = matlab.engine.start_matlab()
        eng.addpath(os.getcwd()+'/MatlabFunctions/',nargout=0)
        print("Producing pickle file : matching radar and lidar acquisition times " + chosenDate)
        
        JoinedData = defaultdict(list);

        for indice, k in enumerate(filepaths_radar):    
            for beam_number in range(1,11):
                if radarStart[beam_number][indice].minute>6:
                    # Match times
                    index1 = matchTimes(radarStart[beam_number][indice], 
                                        radarEnd[beam_number][indice],lidarStart[radar2lidar[str(beam_number)]],
                                        lidarEnd[radar2lidar[str(beam_number)]], len(filepaths_radar))           
                    # Match beams
                    beam_lidar = radar2lidar[str(beam_number)]
                    target = high_mode-490 if (beam_number >=1 and beam_number <=5) else low_mode-490 

                    # Lidar data
                    interp=[]
                    for kk in range(0, len(index1)):
                        lidar_file = filepaths_lidar[index1[kk].astype(int)]
                        group_name = list(netCDF4.Dataset(lidar_file).groups.keys())[1]
                        sw = xr.open_dataset(lidar_file, group=group_name,decode_times=False)
                        rangeLidar = sw.measurement_height.values[beam_lidar] 
                        #Vertical distance normal to the ground, between the instrument and the center of each range gate.
                        sw.radial_wind_speed_status.values[sw.wind_speed_status.values == 0] = np.nan
                        radial = sw.radial_wind_speed.values[beam_lidar]*sw.radial_wind_speed_status.values[beam_lidar]            
                        #Interpolate radial velocity
                        interp.append(np.interp(target, rangeLidar, radial))
                        del sw   
                    if len(interp)==0: interp = np.full([2,55], np.nan)
                else:
                    interp = np.full([2,55], np.nan)

                JoinedData[beam_number].append(np.nanmean(interp, axis=0))
                out = eng.loadMoments(k, beam_number,nargout=5)
                JoinedData[str(beam_number)+'MomentVel'].append(np.array(out[0]))
                JoinedData[str(beam_number)+'MomentSig'].append(np.array(out[1]))
                JoinedData[str(beam_number)+'MomentNoise'].append(np.array(out[2]))
                JoinedData[str(beam_number)+'MomentSkew'].append(np.array(out[3]))
                JoinedData[str(beam_number)+'MomentWidth'].append(np.array(out[4]))
                                               
                if beam_number == 1:
                    JoinedData['filename'].append(k) # store filenames only once for all beams

        with open(os.getcwd()+'/Lists/'+chosenDate+"JoinedData.pickle", "wb") as f:
            pickle.dump(JoinedData, f)

    return JoinedData

def MatchAverages(df_radialHM, df_radialLM, JoinedData,date_radardt, mode):
    
    """
    Radar output product u,v,w is averaged within a sliding window. 
    This function allows to average lidar radial speed accordingly.
    
    Args:
        df_radialHM, df_radialLM : dataframe with added radar radial velocities and frequency 
                                   output of function uvw2radial (ThreeDWindRadarFunctions.py)
        JoinedData : dictionary of lidar radial speed for each spectrogram (= each radar file, all beams, all gates)
                     output of function retrieveLidarRadialSpeed 
        date_radardt : list of datetime for the day chosen, as encoded in the name of the files
                       corresponding to beginning of each measurement cycle
                       output of function loadRadar (LoadFunctions.py)
        mode : 'high' or 'low' depending on your wish
    Returns:
        CoherencyData : dictionary of matched average measurements for all beams
    """ 

    r = [[1,6] if mode == 'high' else [6,11]]
    dfTemp = df_radialHM.copy() if mode == 'high' else df_radialLM.copy()
    dfTemp.replace(0.0, np.nan, inplace=True) #Yes, no ??

    listStart = list(dfTemp['Time_start'].drop_duplicates());listEnd = list(dfTemp['Time_end'].drop_duplicates())
    indexSplit = np.arange(0,len(dfTemp)+55, 55)
    CoherencyData=defaultdict(list);

    for indice in range(0,len(listStart)):
        mask=[]
        for k in date_radardt:
            mask.append(k > listStart[indice] and k <listEnd[indice])
        mask=pd.Series(mask)
        index = mask[mask==True].index.values

        for beam_number in range(r[0][0],r[0][1]):
            CoherencyData[str(beam_number)+'speedR'].append(np.array(dfTemp.iloc[indexSplit[indice]:indexSplit[indice+1]][str(beam_number)+'speedR']))

            if index.size >1:
                CoherencyData[str(beam_number)+'speedL'].append(np.nanmean(JoinedData[beam_number][index[0]:index[-1]], axis=0))
            else:
                CoherencyData[str(beam_number)+'speedL'].append(np.full(55, np.nan))
        CoherencyData['Time_end'].append(listEnd[indice])
        
    return CoherencyData

def MatchRadarTimes(df_radial, beam_number, gate, mode, index, radarStart, radarEnd):
    """
    Radar output u,v,w was projected onto the beam direction in order to obtain radial velocity and frequency.  
    The end time of this output is matched with each radar 30 seconds acquisition within the time interval chosen
    The only aim is to be able to plot it on figure from code 4_Visualize. 
    Args: 
        df_radial : dataframe containing 3D wind measurements (u,v,w,direction,altitude,etc)
                    with added radial velocities and frequency
                    output of function uvw2radial (ThreeDWindRadarFunctions.py)
        beam_number : radar beam identification number 
        gate : altitude identification index (0 to 54) 
        mode : 'high' or 'low' depending on your wish
        index : of radar files that are within the time interval chosen
                output of function loadDataPlot (LoadFunctions.py)
        radarStart, radarEnd : dictionaries of radar start and end acquisition times for each beam
                               output of function radarStartEnd
    Returns:
        radarF : numpy array of Doppler radar frequency shifts
    """ 
    try:
        radarF=np.array([])
        for k in index:
            df_radar = df_radial[df_radial[:]['Altitude [m, ASL]'].apply(pd.to_numeric) == mode[gate]]
            df_radar = df_radar[(df_radar['Time_end'] >= radarEnd[beam_number][k]) &
                                (df_radar['Time_end'].shift(1) <= radarStart[beam_number][k])] 
            radarF = np.append(radarF, df_radar[str(beam_number)+'frequencyR'].item())
    except ValueError:
        radarF = np.full(len(index), np.nan)        
    
    return radarF

def MatchClosestMoment(Lidar, Radar):
    """
    The radar raw files give 3 sdetected peaks but no indication on which on will be chosen in the end.
    The function favors the peak whose value is the closest to the lidar Doppler velocity shift.
    Args: 
        Lidar : Doppler frequency shifts 
        Radar :
    Returns:
       result: numpy array of Doppler radar frequency shifts
    """     
    indexMoment = np.argmin(np.vstack([np.abs(Lidar-f2vRadar(Radar[0])),
                       np.abs(Lidar-f2vRadar(Radar[1])),
                       np.abs(Lidar-f2vRadar(Radar[2]))]),
            axis=0)
    indexMomentoneHot = np.zeros((3, 55))
    indexMomentoneHot[indexMoment,np.arange(indexMoment.size)] = 1
    maX = np.amax(indexMomentoneHot*Radar, axis=0)
    miN = np.amin(indexMomentoneHot*Radar, axis=0)
    miN[miN == 0] =1; maX[maX == 0] = 1; result = maX*miN 
    return result


