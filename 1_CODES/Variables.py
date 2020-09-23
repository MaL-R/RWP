import numpy as np

""" Altitudes of low and high mode radar
Units : Meters above sea level, vertical distance normal to ground
Center of resolution volume
Retrieve 490meter if you want them above ground level 
"""

low_mode = np.array([ 596,  653,  711,  768,  825,  883,  940,  997, 1055, 1112, 1169,
       1227, 1284, 1341, 1399, 1456, 1513, 1571, 1628, 1685, 1743, 1800,
       1857, 1915, 1972, 2029, 2087, 2144, 2201, 2259, 2316, 2373, 2431,
       2488, 2545, 2603, 2660, 2717, 2775, 2832, 2889, 2947, 3004, 3061,
       3119, 3176, 3233, 3291, 3348, 3405, 3463, 3520, 3578, 3635, 3692])

high_mode = np.array([ 789,  933, 1076, 1219, 1363, 1506, 1649, 1793, 1936, 2079, 2223,
       2366, 2509, 2653, 2796, 2939, 3083, 3226, 3370, 3513, 3656, 3800,
       3943, 4086, 4230, 4373, 4516, 4660, 4803, 4946, 5090, 5233, 5376,
       5520, 5663, 5806, 5950, 6093, 6236, 6380, 6523, 6666, 6810, 6953,
       7096, 7240, 7383, 7527, 7670, 7813, 7957, 8100, 8243, 8387, 8530])

lidar_mode = np.array([  200,   300,   400,   500,   600,   700,   800,   900,  1000,
        1100,  1200,  1300,  1400,  1500,  1600,  1700,  1800,  1900,
        2000,  2100,  2200,  2300,  2400,  2500,  2600,  2700,  2800,
        2900,  3000,  3100,  3200,  3300,  3400,  3500,  3600,  3700,
        3800,  3900,  4000,  4100,  4200,  4300,  4400,  4500,  4600,
        4700,  4800,  4900,  5000,  5100,  5200,  5300,  5400,  5500,
        5600,  5700,  5800,  5900,  6000,  6100,  6200,  6300,  6400,
        6500,  6600,  6700,  6800,  6900,  7000,  7100,  7200,  7300,
        7400,  7500,  7600,  7700,  7800,  7900,  8000,  8100,  8200,
        8300,  8400,  8500,  8600,  8700,  8800,  8900,  9000,  9100,
        9200,  9300,  9400,  9500,  9600,  9700,  9800,  9900, 10000,
       10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900,
       11000, 11100, 11200, 11300, 11400, 11500, 11600, 11700, 11800,
       11900, 12000]) 


""" Radar beams names
5 beams in high mode, then switching to low mode
"""
beam_radar_pretty = {'1': 'Beam 1 (vertical) of high mode', 
                   '2':'Beam 2 (northwest) of high mode',
                   '3':'Beam 3 (southeast) of high mode',
                   '4':'Beam 4 (northeast) of high mode',
                   '5':'Beam 5 (southwest) of high mode',
                   '6':'Beam 6 (vertical) of low mode',
                   '7':'Beam 7 (northwest) of lowmode',
                   '8':'Beam 8 (southeast) of low mode',
                   '9':'Beam 9 (northeast) of low mode',
                   '10':'Beam 10 (southwest) of low mode'};



""" Radar beams azimuths
Units : Degrees
Vertical beam has no azimuth. Two cycles corresponding to high and low mode measurements
Perfoms diagonal measurements to have closest posible opposite beams measurements in time.
In Matlab, index starts at 1 so index beam range is 1 to 10
"""
beam_radar = {'1':np.NaN, '2':315, '3':135, '4':45, '5':225, '6':np.NaN, '7':315, '8':135, '9':45, '10':225 }



""" Correspondance lidar and radar beams
Units : beams' number as in file
Configuration of July 2020, that is lidar with a 315° offset meaning lidar and radar beams aligned
Lidar performs a full circle and not diagonal measurements
In Python, index starts at 0 so from 0 to 4
"""
radar2lidar = {'1':4, '2':0, '3':2, '4':1, '5':3, '6':4, '7':0, '8':2, '9':1, '10':3}
lidar2radar = {4:1,0:2,2:3,1:4,3:5}



""" Constants for deep learning
The numbers 600 and 544 refer to frequency (axis 0, y-axis) and acquisition time (axis 1, x-axis) respectively
Classes : (0) no visible wind, (1) visible wind only, (2) massive bird contamination and (3) slight contamination and wind still visible. 
"""
INPUT_WIDTH = 544
INPUT_HEIGHT = 600
NUMBER_OF_CLASSES = 4



""" Lists of days in the full dataset, classified as rainy if it rained at least once (ground-based measurements)
Note that the Github only inclues samples of the raw data
"""
dateListRainy = ['2020619','2020627','2020628','2020629','2020710','2020711','2020715','2020716','2020721','2020722','2020724','2020801','2020802','2020803','2020804','2020810','2020813','2020814','2020816','2020817']

dateListSunny = ['2020620','2020621','2020622','2020623','2020624','2020625','2020626','2020630','2020701','2020702','2020703','2020704','2020705','2020706','2020707','2020708','2020709','2020712','2020713','2020714','2020717','2020718','2020719','2020720','2020723','2020725','2020726','2020727','2020728','2020729','2020730','2020731','2020805','2020806','2020807','2020808','2020809','2020811','2020812','2020815']



""" Manual classification and visualisation
Hourly temporal split of the day. 
Can be modified directly in the script to chose specific hours or longer time series.
"""
start = ['0000', '0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300']
end = ['0100', '0200', '0300', '0400', '0500', '0600', '0700', '0800', '0900', '1000', '1100', '1200', '1300', '1400', '1500', '1600', '1700', '1800', '1900', '2000', '2100', '2200', '2300', '2359']

