import json
import fitter
import pylab
import array
import visa
import pyvisa
import time
import pylab
import array
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from numpy import *
from lmfit import Model
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.signal import find_peaks



# Load data
with open("X_A_6_257-259-261-263-265-267-269-271_20211023T090545/amplitudeData.json", "r") as datafile:
    data = json.load(datafile)

# Plotting one of the 8 traces in this scan
dwaChan = '6' # This can be 0 through 7
freq = data[dwaChan]['freq']
ampl = data[dwaChan]['ampl']
baseline = savgol_filter(ampl, 61, 5)
baseline_sub = ampl - baseline
baselineafter = savgol_filter(baseline_sub, 15, 3)

Output = []
Frequency = []

Output = baselineafter
Frequency = freq
x = np.linspace(84, 87, 50) 

#For plotting a continuous fit function
#x = baselineafter   

y_fit = []
y_ifit = []


#Fit Parameters
# par[0] = c1, par[1] = c2, par[2] = Gamma, par[3] = resonant frequency
parameters = []

#Initial guess for fit parameters
par = [0.5,0.5,0.005,86]

#Fits the output amplitude vs frequency waveform from the oscilloscope
fitter.bipolar_reso(Output, Frequency, par, y_fit, y_ifit, parameters)


mu = 1.6*(10**(-4))          #Mass density of the Copper-Beryllium Wire
omega = parameters[3]        #Resonant Frequency
wire_len = 6                 #Length of the Wire   

tension = 4*mu*(omega*wire_len/(2*np.pi))**2
print(tension)


pylab.plot(Frequency, Output, '-', label='Data')
pylab.plot(x, fitter.resonance(x, parameters[0] , parameters[1], parameters[2], parameters[3]), 'r', label='Fit Function')
pylab.legend(loc='upper left')
pylab.title('Frequency Spectrum')
pylab.xlabel('Frequency (Hz)')
pylab.ylabel('Amplitude (V)')
pylab.show()

