function [MomentVel, MomentSig, MomentNoise, MomentSkew, MomentWidth] = loadMoments(filename,index_beam)

% Load file
Data = loadfile(filename);

% Retrieve moments
MomentVel = Data(index_beam).Moments.Vel;
MomentSig = Data(index_beam).Moments.Sig;
MomentNoise = Data(index_beam).Moments.Noise;
MomentSkew = Data(index_beam).Moments.Skew;
MomentWidth = Data(index_beam).Moments.Width;




