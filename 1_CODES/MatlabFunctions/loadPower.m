function [max1,max2,max3,max4,max5,width1,width2,width3,width4,width5] = loadPower(filename)

% Load file
Data = loadfile(filename);

% Do fft
max1 = max(Data(1).Spectra)';
max2 = max(Data(2).Spectra)';
max3 = max(Data(3).Spectra)';
max4 = max(Data(4).Spectra)';
max5 = max(Data(5).Spectra)';
width1 = Data(1).Moments.Width(1,:)';
width2 = Data(2).Moments.Width(1,:)';
width3 = Data(3).Moments.Width(1,:)';
width4 = Data(4).Moments.Width(1,:)';
width5 = Data(5).Moments.Width(1,:)';

