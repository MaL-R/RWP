function [timeSerieR, timeSerieI, spec, fspec, tspec, spectra, fs] = loadSpec(filename,index_beam, index_gate)

% spectrograms NOT converted to dB 

% Load file
Data = loadfile(filename);

% Do fft

timeSerie = Data(index_beam).TimeSeries(:,index_gate);
timeSerieR = real(timeSerie);timeSerieI = imag(timeSerie);
spectra = Data(index_beam).Spectra(:,index_gate);

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

tspec=(0:N-1)*a;
tspec=tspec/fs;

fspec=[-1+2/M, 1];
fspec=fspec*fs/2;
fspec=linspace(fspec(1),fspec(2),M);

ts=(0:(size(timeSerie)-1))/fs;

% Move zero frequency to the center and Nyquest frequency to the top.
spec=circshift(spec,M/2-1);

% Convert to Db
% spec=20*log10(abs(spec)+realmin);

fspec = fspec.';
tspec=tspec.';

