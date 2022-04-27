import resonance_fitting
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
with open("data/X_A_3_97-99-101-103-105-107-109-111_20211023T041622/amplitudeData.json", "r") as datafile:
    data = json.load(datafile)

# Plotting one of the 8 traces in this scan
dwaChan = '1' # This can be 0 through 7
freq = data[dwaChan]['freq']
ampl = data[dwaChan]['ampl']
roundex=[]
expected_resonances = roundex
 
bsub = resonance_fitting.baseline_subtracted(np.cumsum(ampl))
bsubabs = np.abs(bsub)
smooth = savgol_filter(bsubabs, 51, 3)

        

pks, _ = find_peaks(smooth,prominence=5)
fpks = np.array([freq[pk] for pk in pks])

'''
placements, costs, diffs, tensions = resonance_fitting.analyze_res_placement(freq,smooth,expected_resonances,fpks)
print("peaks: ",fpks)

sorted_placements = np.array([x for _, x in sorted(zip(costs, placements))])
sorted_diffs = np.array([x for _, x in sorted(zip(costs, diffs))])
sorted_tensions = np.array([x for _, x in sorted(zip(costs, tensions))])
sorted_costs = np.array([x for _, x in sorted(zip(costs, costs))])

lowest_cost = sorted_costs[0]
lowest_placement = sorted_placements[0]

print("lowest: ",lowest_placement)
print("sorted costs: ",sorted_costs[:3]) 
print("sorted placements: ",sorted_placements[:3]) 
select_best = (sorted_costs < 1.2*lowest_cost)
best_tensions = sorted_tensions[select_best]
best_tensions_std = np.std(best_tensions,0)
print("best std: ",best_tensions_std)


print("best std: ",sorted_tensions)
'''

x = smooth
ll = np.array(freq)
peaks, properties = find_peaks(x, height=20, width=1)

properties["prominences"], properties["widths"]
(array([5, 99]), array([0.01 ,5]))

plt.plot(ll,x)

plt.plot(ll[peaks], x[peaks], "x")

print('peaks',fpks)
print('peaks2',ll[peaks])

mu = 1.6*(10**(-4))          #Mass density of the Copper-Beryllium Wire
omega = fpks[0]       #Resonant Frequency  
wire_len = 1.28238                 #Length of the Wire 

tension = 4*mu*(omega*wire_len)**2
print('tension = ', tension)



plt.show()