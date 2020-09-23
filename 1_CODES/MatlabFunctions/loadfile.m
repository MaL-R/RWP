function Measure = loadfile(filename)

Measure = pca_read(filename);

nmesure=length(Measure);
% rendre plus lisible les entetes:
for i=1:nmesure
    Measure(i).HEADER = extract_header(Measure(i).Header);
end
