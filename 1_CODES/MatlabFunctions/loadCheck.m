function x = loadtime(filename)

Measure = pca_read(filename);
if length(Measure) <10 
    x=0;
else
    x=1;
end