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
import resonance_fitting



# Load data
with open("data/X_A_6_257-259-261-263-265-267-269-271_20211023T090545/amplitudeData.json", "r") as datafile:
    data = json.load(datafile)

# Plotting one of the 8 traces in this scan
dwaChan = '2' # This can be 0 through 7
freq = data[dwaChan]['freq']
ampl = data[dwaChan]['ampl']



# Subtracting the baseline
def baseline_subtracted(amps):
    '''Use the savgol filter to smooth out a trace.'''
    smooth_curve = savgol_filter(amps, 111, 3)
    return savgol_filter(amps-smooth_curve, 7, 2)

bsub = baseline_subtracted(np.cumsum(ampl))
bsubabs = np.abs(bsub)
smooth = savgol_filter(bsubabs, 51, 3)


x = smooth
ll = np.array(freq)
peaks,_ = find_peaks(x, prominence=15)


plt.plot(ll,x)

plt.plot(ll[peaks], x[peaks], "x")

print('peaks',ll[peaks])
 
'''
mu = 1.6*(10**(-4))                #Mass density of the Copper-Beryllium Wire
omega = peaks[0]                   #Resonant Frequency  
wire_len = 1.28238                 #Length of the Wire   

tension = 4*mu*(omega*wire_len)**2
print('tension = ', tension)
'''
plt.title("smooth")
plt.plot(freq, smooth)
pylab.legend(loc='upper left')
pylab.title('Frequency Spectrum')
pylab.xlabel('Frequency (Hz)')
pylab.ylabel('Amplitude (V)')
pylab.show()

