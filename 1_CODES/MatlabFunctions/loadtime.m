function [Year,Month,Day,Hour,Minute,Second] = loadtime(filename, index_beam)

Measure = pca_read(filename);

% année de la mesure                 
Year=Measure(index_beam).Header(16);
% mois de la mesure
Month=Measure(index_beam).Header(22);
% jour de la mesure
Day=Measure(index_beam).Header(21);
% heure de la mesure
Hour=Measure(index_beam).Header(18);
% minute de la mesure
Minute=Measure(index_beam).Header(19);
% seconde de la mesure
Second=Measure(index_beam).Header(20);
