# RWP
A Pulse-Doppler Radar Wind Profiler (RWP) is an active remote sensing instrument used in meteorology whose output product is a 3D wind field. A new method to retrieve radial wind velocity from spectrograms using Convolutional Neural Networks is introduced. The collection of data is provided by the Federal Office of Meteorology and Climatology MeteoSwiss and collected in Payerne, Switzerland. It covers the summer months of the year 2020. Only a subset is here provided and shows the folder architectur required for the codes to run.

As a first step, spectrograms are split into different classes as follows: (0) no visible/measured wind, (1) visible wind only, (2) massive bird contamination (3) slight contamination and wind still visible. In terms of accuracy, precision and recall, the model achieves a solid performance of 94\% on the test set with a tendency to mix classes 1/3 and 2/3. 

Spectrograms either too heavily contaminated (class 2) or lacking a wind signal (class 0) are discarded in a second phase. A Doppler lidar provides the radial velocity for each spectrogram. Accross the test set, a R² of 0.97 is obtained along with a mean absolute error of 0.22 m/s.

A detailed description of the project can be found in the report in the repository.

## Install matlab engine
Follow instructions from https://www.mathworks.com/help/matlab/matlab-engine-for-python.html. It requires administrator rights.

## Setting up an environment with pyenv

Install a python version if needed
```pyenv install 3.6.9```
Create a new python environment based on the python version
```pyenv virtaulenv RWPenv 3.6.9```
Activate it
```
pyenv activate RWPenv
```
Install required packages 
```
pip install –r requirements_pyenv.txt
```
Clone the github
```
cd /path/to/somewhere/
git clone https://github.com/MaL-R/RWP/
```
Set your local default environment
```
cd RWP
pyenv local RWPenv
```
The package Tkinter is necessary for the manual classification task and runs on C. As pyenv only manages python dependencies, it proved to be complex to install. However, as a last possibility it would be possible to use the system Python.

## Setting up the environment with anaconda
```
conda create --name RWPenv python=3.6.9
conda activate RWPenv
conda install --file requirements_conda.txt
```
Tkinter comes with initial creation of the environement, version tk 8.6.8, as anaconda also manages C dependencies. 

## Codes
```1_Barbs&Rain.ipynb``` allows to visualise wind barbs and ground-based rain measurement on a chosen day. Raw data for days July 11 and 12, 2020 are provided.
```2_ProduceLists.ipynb```: Doppler radar and lidar measurements are matched. As it takes some time, the output default dictionaries are saved in pickles files. The original dataset begins on June 19 and ends on August 18, 2020. Seven days are missing in this period as the radar had to be turned off for technical reasons. All pickles files are provided.
```3_CoherencyAnalysis.ipynb``` and ```3_CoherencyAnalysisFixed``` perform an assessment of the radar and lidar correlations, in the form of barplots and scatterplots. Different filters are established in order to quantify the influence of precipitation, bird contamination, temporal and spatial resolutions.
```4_Visualization.ipynb``` outputs a figure with spectrograms, average spectra, time serie and wind barbs. Both the average radar wind vectors converted back to radial Doppler frequency shifts and the lidar shift are overlapped on the spectrograms.
```5_ManualClassification.ipynb```: interactive tool which allows the task of manual classification. The code makes use of package Tkinter aforementionned.
```6_Classification.ipynb```: TensorFlow model and development for the classification. All results and figures can still be viewed on the notebook. 
```7_Regression.ipynb```: TensorFlow model and development for the regression. All results and figures can still be viewed on the notebook. 
