function [spec,MomentNoise, maxTimeSerie,fs] = loadSpecAdditional(filename,index_beam, index_gate)

% Load file
Data = loadfile(filename);

% Retrieve Moments
MomentVel = Data(index_beam).Moments.Vel(1,index_gate);
MomentSig = Data(index_beam).Moments.Sig(1,index_gate);
MomentNoise = Data(index_beam).Moments.Noise(1,index_gate);
MomentSkew = Data(index_beam).Moments.Skew(1,index_gate);
MomentWidth = Data(index_beam).Moments.Width(1,index_gate);

% Do fft
timeSerie = Data(index_beam).TimeSeries(:,index_gate);
timeSerieR = real(timeSerie);timeSerieI = imag(timeSerie);
maxTimeSerie = max(max(abs(timeSerieR)),max(abs(timeSerieI)));
%spectra = Data(index_beam).Spectra(:,index_gate);

% Récuperer la fréquence d'echantillonage
IPP_s = double(Data(index_beam).Header(8))*double(Data(index_beam).Header(33));
nb_IPP = double(Data(index_beam).Header(42));
nci = double(Data(index_beam).Header(7))*double(Data(index_beam).Header(64));
fs = 1.0e9/(nci * IPP_s * nb_IPP);

Ls=length(timeSerie);
kv.xres = 800;
kv.yres = 600;
kv.tfr = 1;
flags.norm = '2';

[a,M,L,N,Ndisp]=gabimagepars(Ls,kv.xres,kv.yres);

% Discrete Gabor transform
G={'gauss',kv.tfr,flags.norm};
spec=dgt(timeSerie,G,a,M);
spec = abs(spec);

% Cut away zero-extension.
spec=spec(:,1:Ndisp);

M=size(spec,1);
N=size(spec,2);

% Move zero frequency to the center and Nyquest frequency to the top.
spec=circshift(spec,M/2-1);

% Convert to Db
spec=20*log10(abs(spec)+realmin);
spec(spec>30)=30;
spec(spec<0)=0;

