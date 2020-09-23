import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
from Variables import *
from OtherUsefulFunctions import *
from LoadFunctions import *


def plotAlltogether(chosenDate, start, end, gate, mode, beam_number,
                    radarStart, radarEnd, JoinedData, pathSave,df_radial,data,date_radardt, 
                    optionSave, optionSpec, optionPlot, linewidth=3.0, fontsize1=20, fontsize2=16):
    """
    Allows to visualise all data available together on a certain interval of time
    Parameters can be freely chosen
    
    Args:
        chosenDate : string of day chosen, as for example '2020619' or '2020701'
        start, end : start and end hours, as for example '0001' and '0202'        
        gate : altitude identification index (0 to 54) 
        mode : 'high' or 'low' depending on your wish
        beam_number : radar beam identification number 
        radarStart, radarEnd : dictionaries of radar start and end acquisition times for each beam
                               output of function radarStartEnd (MatchFunctions.py)
        JoinedData : dictionary of lidar radial speed for each spectrogram (= each radar file, all beams, all gates)
             output of function retrieveLidarRadialSpeed (MatchFunctions.py)
        optionSave : if True, will save the plot in the folder specified in pathSave
        optionSpec : optionSpec = 2 will plot a spectrogram with one column out of 2. 
                     Up to 10 it does not change the visualization of the wind but allows a much faster plotting
        optionPlot : =1 only spectrogram
                     =2 spectrogram + average spectra
                     =3 spectrogram + average spectra + time serie
                     =4 spectrogram + average spectra + time serie + wind barb
        fontsize1, fontsize2 : size of the main and smaller fonsizes
        
    """     
    # Gates depending on mode
    modeAltitude = high_mode if mode == 'high' else low_mode

    # Load data
    start_str, end_str,start_date, end_date, timeserieR,timeserieI, spectrogram,fspec, tspec, spec, index, fs, fMin, fMax = loadDataPlot(start, end, chosenDate,beam_number,gate, radarStart, radarEnd, JoinedData,option=optionSpec)
    
    try:
        lidarF = np.array([JoinedData[beam_number][j][gate] for j in index])
    except IndexError:
        lidarF = np.empty(len(index))
        lidarF[:]=np.nan
        
    radarF = MatchRadarTimes(df_radial, beam_number, gate, modeAltitude, index, radarStart, radarEnd)
    date_t = [radarStart[beam_number][j].strftime('%H:%M') for j in index]
    date_toute1 = [dt.datetime.strftime(k, '%H%M') for k in date_radardt][index[0]:index[-1]+1]
    date_toute2 = [dt.datetime.strftime(k, '%H:%M') for k in date_radardt][index[0]:index[-1]+1]

    # Colormap & fontsize
    light_jet = cmap_map(lambda x: x + 0.3, matplotlib.cm.jet) #allows to have transparency on the jet colormap
    matplotlib.rcParams.update({'font.size': fontsize1}) 

    # Figure
    fig = plt.figure()
    fig.suptitle('Gate ' + str(gate) + ' : ' + str(modeAltitude[gate]) + 'm, ASL' 
                 + '\nDate : ' + padChosenDate(chosenDate)
                 + '\nBeam : ' + str(beam_number))

    try:
        outer = gridspec.GridSpec(ncols=40, nrows=optionPlot, figure=fig, hspace=.3, left=0.1)
    except TypeError: # matplotlib workstation 2.1.1 : no argument figure for gridspec
        outer = gridspec.GridSpec(ncols=40, nrows=optionPlot, hspace=.3, left=0.1)
        
    if optionPlot>=1:
        inner1 = gridspec.GridSpecFromSubplotSpec(1, len(spec), subplot_spec=outer[0,:-1], wspace=0.)
    if optionPlot>=2:
        inner2 = gridspec.GridSpecFromSubplotSpec(1, len(spec), subplot_spec=outer[1,:-1], wspace=0.)
        specMin, specMax, yy, dataFilt, yyMin, yyMax = plot_spectra(spec,fs, Filt=0)

    for indice, dataSpec in enumerate(spec):

        if optionPlot >=1:# only spectrogram
            ax1 = fig.add_subplot(inner1[indice])
            im1 = ax1.pcolormesh(tspec[indice], fspec[indice], toDb(spectrogram[indice]),
                                 cmap=light_jet, vmin=0, vmax=30);
            lineRadar = ax1.axhline(radarF[indice], color='fuchsia',linestyle='dashed',linewidth=linewidth)
            lineLidar = ax1.axhline(v2fRadar(lidarF[indice]), color='black',linestyle='dashed',linewidth=linewidth)

            ax5 = fig.add_subplot(outer[0,-1])
            cbar1 = fig.colorbar(im1, cax=ax5)
            cbar1.set_label('Power [dB]', rotation=90)

            ax1.set_ylim([np.max(fMin), np.min(fMax)]) 
            plt.setp(ax1.get_xticklabels(), visible=False);
            ax1.xaxis.set_ticks_position('none')        
            if not ax1.is_first_col():
                ax1.set_xlabel(date_toute2[indice])
                plt.setp(ax1.get_yticklabels(), visible=False);
            if ax1.is_first_col():
                ax1.set_ylabel('D. freq. shift [Hz]');
                ax1.set_xlabel('Beginning of \nacquisition \n[Time UTC]', fontsize=fontsize2);

        if optionPlot>=2:# spectrogram + average spectra
            vel = JoinedData[str(beam_number)+'MomentVel'][index[indice]][:,gate]
            width = JoinedData[str(beam_number)+'MomentWidth'][index[indice]][:,gate]
            noise = JoinedData[str(beam_number)+'MomentNoise'][index[indice]][:,gate]
            ax2 = fig.add_subplot(inner2[indice])
            lineSpectra = ax2.semilogx(dataSpec, yy[indice]) #post processing

            ax2.axhline(0, color='grey', alpha=0.5)
            lineSnr = ax2.axvline(noise, color='g')
            ax2.axhline(radarF[indice], color='fuchsia',linestyle='dashed')
            ax2.axhline(v2fRadar(lidarF[indice]), color='black',linestyle='dashed')

            xInterp = max_nearest(yy[indice], vel, dataFilt[indice])
            lineMoment = ax2.errorbar(xInterp, vel, yerr=width, fmt='o', color='tab:orange')

            ax2.set_xlim([np.min(specMin), np.max(specMax)]) 
            ax2.set_ylim([np.max(yyMin), np.min(yyMax)])

            locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
            ax2.xaxis.set_minor_locator(locmin)
            ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax2.grid(axis='x', which='major', linewidth=1)
            ax2.grid(axis='x', which='minor', alpha=0.4)

            if not ax2.is_first_col():
                plt.setp(ax2.get_yticklabels(), visible=False);
                for tick in ax2.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fontsize2)
            if ax2.is_first_col():
                ax2.set_ylabel('D. freq. shift [Hz]');
                plt.setp(ax2.get_xticklabels(), visible=False);
                ax2.xaxis.set_ticks_position('none') 
                ax2.set_xlabel('Power [dB]', fontsize=fontsize2);

        if optionPlot>=3: # spectrogram + average spectra + time serie
            ax3 = fig.add_subplot(outer[2,:-1])
            plot_timeserie(ax3, timeserieR, timeserieI, spectrogram, date_toute2,fontsize2);
            h,l=ax3.get_legend_handles_labels()
            ax3.legend(np.unique(l),fontsize=fontsize2, loc='upper left', ncol=2)

        if optionPlot>=4: # spectrogram + average spectra + time serie + wind barb
            ax4 = fig.add_subplot(outer[3,:-1])
            im2, newcmp, bounds, norm = plot_barb_filter(fig, ax4, data, gate, modeAltitude, start_str, end_str);

            ax6 = fig.add_subplot(outer[3,-1])
            plot_barb_colorbar(im2, ax6, newcmp, bounds, norm)

    if optionPlot ==1:
        legend = fig.legend((lineRadar, lineLidar, lineSpectra[0]), 
                        ('Radar radial Doppler \nfrequency shift','Lidar radial Doppler \nfrequency shift'), 
                        loc='upper left', ncol=2)
    if optionPlot >= 2:
        legend = fig.legend((lineRadar, lineLidar, lineSpectra[0], lineSnr, lineMoment), 
                            ('Radar radial D. freq. shift','Lidar radial D. freq. shift', 
                             'Radar average spectra \n(after treatment)','Signal to noise ratio', 
                             'Radar radial D. freq. shift and \nwidth of 3 highest detected peaks'), 
                            loc='upper left', ncol=2)

    plt.gcf().set_size_inches(32, 16)

    if optionSave is True:
        plt.savefig(pathSave+chosenDate+'_'+str(beam_number)+'_gate'+str(gate)+'_'+start_str[-4:]+'_'+end_str[-4:]) 

    plt.close()
    
    return date_toute1, date_toute2,date_t, fig

def plot_timeserie(ax, timeserieR, timeserieI, spectrogram, date_t,fontsize2):
    varR = timeserieR[0]
    varI = timeserieI[0]
    maxxR = [round(max(np.max(np.abs(varR)), np.max(np.abs(varI))),2)]
    for t in range(1,len(timeserieR)):
        varR = np.concatenate([varR, timeserieR[t]])
        varI = np.concatenate([varI, timeserieI[t]])
        maxxR.append(round(max(np.max(np.abs(timeserieR[t])), np.max(np.abs(timeserieI[t]))),2))
    lineI = ax.plot(varI, color='gold', label='Imaginary component');   
    lineR = ax.plot(varR, color='indigo', label='Real component');    
    ax.set_xlim([0,3264*len(spectrogram)+1])
    ax.set_xticks(np.arange(0, 3264*len(spectrogram)+1, 3264));
    ax.set_xticks(np.arange(3264/2, 3264*len(spectrogram)+1-3264/2, 3264), minor=True);
    ax.set_xticklabels(date_t, minor=True);
    #ax.set_xticklabels(maxxR, minor=True,fontsize=fontsize2);
    ax.set_xticklabels([]);
    ax.xaxis.grid(True,linewidth=1.5, color='grey')
    ax.yaxis.grid(True, alpha=0.4)  


def plot_barb_filter(fig, ax, data ,gate, modeAltitude, start_str, end_str):

    data = data[data['Time_written'] >= start_str]
    data = data[data['Time_written'] <= end_str]

    if gate <= 10:
        data = data[data['Altitude [m, ASL]'] >= modeAltitude[0]]
        data = data[data['Altitude [m, ASL]'] <= modeAltitude[gate+10]]        
    if gate >10 and gate <45: 
        data = data[data['Altitude [m, ASL]'] >= modeAltitude[gate-10]]
        data = data[data['Altitude [m, ASL]'] <= modeAltitude[gate+10]]
    if gate >= 45:
        data = data[data['Altitude [m, ASL]'] >= modeAltitude[gate-10]]
        data = data[data['Altitude [m, ASL]'] <= modeAltitude[len(modeAltitude)-1]]
     
    bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    newcmp = colormap4flag()
    norm = matplotlib.colors.BoundaryNorm(bounds, newcmp.N)
    
    ax.axhline(modeAltitude[gate]/1000, color='grey', alpha=0.2)
    try:
        img = ax.barbs(data['Time_written']-pd.Timedelta(5,'min'), data['Altitude [m, ASL]'].divide(1000),data['East [m/s]'], data['North [m/s]'], data['Speed'], cmap=newcmp,norm=norm, sizes=dict(emptybarb=0.1), fill_empty=True,linewidth =1.5, pivot='middle');
    except ValueError: #-pd.Timedelta does not work on workstation 
        img = ax.barbs(data['Time_written'], data['Altitude [m, ASL]'].divide(1000),data['East [m/s]'], data['North [m/s]'],data['Speed'], cmap=newcmp,norm=norm, sizes=dict(emptybarb=0.1), fill_empty=True,linewidth =1.5, pivot='middle');
    
    ax.set_xlabel('Time UTC');
    ax.set_ylabel('Altitude [km, ASL]');
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.grid(axis='y', which='minor', linestyle=':')

    ax.set_xlim(data['Time_written'].min(), data['Time_written'].max());
    ax.set_ylim(data['Altitude [m, ASL]'].divide(1000).min(), data['Altitude [m, ASL]'].divide(1000).max());
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval = 10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    return img, newcmp, bounds, norm

def plot_barb_colorbar(img, ax, newcmp, bounds, norm):
    """
    
    Args:
        img : 
        ax :
        newcmp, bounds, norm :
    """ 
    cbar = plt.colorbar(img, cax=ax, cmap=newcmp, norm=norm, boundaries=bounds, ticks=bounds)
    cbar.ax.set_yticklabels(['0',' ',' ',' ',' ','5',' ',' ',' ',' ','10',
                             ' ',' ',' ',' ','15',' ',' ',' ',' ',' ','flag'])
    cbar.set_label('Wind speed [m/s]', rotation=90)
    

def plotWindBarbs(chosenDate, data, mode, optionSave, pathSave, optionFilter,limitLow,limitHigh):
    
    paddedDate = padChosenDate(chosenDate)
    start_str = str(paddedDate + '-'+'0000')
    end_str = str(paddedDate + '-'+ '2359')
    data = data[data['Time_written'] >= start_str]
    data = data[data['Time_written'] <= end_str]
    
    data = data.iloc[np.arange(0,data.shape[0],optionFilter),:]
    modeAltitude = high_mode if mode == 'high' else low_mode
    bounds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,100,101]
    newcmp = colormap4flag()
    norm = matplotlib.colors.BoundaryNorm(bounds, newcmp.N)
    
    fig = plt.subplots();
    plt.gca().set_title('Date : '+paddedDate[0:4]+'.'+paddedDate[4:6]+'.'+paddedDate[6:8])
    img = plt.barbs(data['Time_written'], data['Altitude [m, ASL]'].divide(1000),data['East [m/s]'], 
                    data['North [m/s]'],data['Speed'], cmap=newcmp,norm=norm, 
                    sizes=dict(emptybarb=0.1), fill_empty=True,linewidth =1.0, pivot='middle');
             #sizes=dict(emptybarb=0.2, spacing=1e-10, height=0.5), length=6, pivot='middle', 
    
    plt.xlabel('Time UTC');
    plt.ylabel('Altitude [km, ASL]');

    plt.gca().set_xlim(data['Time_written'].min(), data['Time_written'].max());
    plt.gca().set_ylim(limitLow-0.1, limitHigh);
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval = 2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45);
    
    cbar = plt.colorbar(img, cmap=newcmp, norm=norm, boundaries=bounds, ticks=bounds)
    cbar.ax.set_yticklabels(['0','1','2','3','4','5','6','7','8','9','10',
                             '11','12','13','14','15','16','17','18','19','20','>20','flag'])

    cbar.set_label('Wind speed [m/s]', rotation=90)
    
    if optionSave is True:
        plt.savefig(pathSave+chosenDate+'_windBarbs', bbox_inches='tight') 
        plt.close()
        print('Saving wind barbs for date : '+chosenDate)
    

def plot_spectra(spec, fs, Filt): 
    specMax = [];specMin=[]; yy=[]; dataFilt=[];specLidarMax=[];yyMin=[];yyMax=[]

    for indice, data in enumerate(spec):
        specMax.append(np.max(data)); specMin.append(np.min(data))

        nfft = len(data)
        yyTemp= np.arange(-nfft/2, nfft/2, 1)
        yyMin.append(np.min(yyTemp)) ; yyMax.append(np.max(yyTemp))          
        yy.append(yyTemp * fs[indice]/nfft)
        if Filt==1:
            dataFiltTemp = savgol_filter(data,15, 5) #window_length, polyorder
            dataFiltTemp[dataFiltTemp<0]=np.mean(dataFiltTemp)
            dataFilt.append(dataFiltTemp)
        else:
            dataFilt.append(data)
             
    return specMin, specMax, yy, dataFilt, yyMin, yyMax

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
        radarF : numpy array of radar radial Doppler frequency shifts
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

def colormap4flag():
    """
    Colormaps adapted to plot wind barbs 
    Flagged values were set to 100 and will be displayed in pink
    """ 
    viridis = matplotlib.cm.get_cmap('jet', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    pink = np.array([248/256, 24/256, 148/256, 1])
    newcolors[-12:, :] = pink
    return matplotlib.colors.ListedColormap(newcolors)

def cmap_map(function, cmap):
    """ 
    Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    Source : https://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
    Args:
        function, cmap
    Returns: 
        colormap
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector
    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

def interpolate_nearest(array1, value, array2):
    """
    The given radar Doppler shifts of the 3 selected peaks needs to be located on the power axis of the average spectra
    The function allows an interpolation of its position
    Args:
        array1 : frequency 
        value : radar Doppler frequency shift for selected peaks
        array2 : power
    Returns:
        xInterp : interpolated power corresponding to the radar Doppler frequency shift of selected peak
    """ 
    xInterp = [];
    for j in range(0,len(value)):
        idx1 = (np.abs(array1 - value[j])).argmin()
        array1bis = array1.copy()
        array1bis[idx1]=0
        idx2 = (np.abs(array1bis - value[j])).argmin()
        idxMin = min(idx1,idx2) ; idxMax = max(idx1,idx2)
        xInterp.append(np.interp(value[j], [array1[idxMin], array1[idxMax]], [array2[idxMin], array2[idxMax]]))
    return xInterp


def max_nearest(array1, value, array2):
    """
    The given radar Doppler shifts of the 3 selected peaks needs to be located on the power axis of the average spectra
    The function allows associating it to the closest maximum power value
    Args:
        array1 : frequency 
        value : radar Doppler frequency shift for selected peaks
        array2 : power
    Returns:
        xMax : power corresponding to the radar Doppler frequency shift of selected peak
    """ 
    xMax = []
    for j in value:
        idx1 = (np.abs(array1 - j)).argmin()
        array1bis = array1.copy()
        array1bis[idx1]=0
        idx2 = (np.abs(array1bis - j)).argmin()
        xMax.append(max(array2[idx1], array2[idx2]))
    return xMax
