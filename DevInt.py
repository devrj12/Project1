#!/usr/bin/env python
# coding: utf-8

# In[5]:


from scipy.io import loadmat
import os
os.chdir("/home/dev/Desktop/training2017/")

# load the ECG file
path   = os.getcwd()
x      = loadmat('A00001.mat')
import numpy as np
a      = np.array(x)

# check
lon    = x['val']
print(lon)


# In[2]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


xx = lon[0]


# In[13]:


plt.plot(xx)
#plt.show


# In[14]:


# So, this is the plot of the ECG. The next step in calculating the heart rate will be to find the distance 
# between two certain peaks and divide the distance by the time involved ( which can be achieved by the frequency
# involved : 300 Hz as given in https://physionet.org/content/challenge-2017/1.0.0/)


# In[15]:


# On the second part, calculating breath rate ( which I take as respiratory rate as well as is mentioned in the 
# literature) inolves multiple steps. And there are many (hundreds ?) appraoches in the literature discussed on 
# this. So, instead of me trying to write a new algorithm myself, I would seek to change an exisitng algorithm
# to suit my problem.

# The following link has one such algorithm :

# https://gist.github.com/raphaelvallat/55624e2eb93064ae57098dd96f259611

"""
I copy the algorithm below which needs to be very carefully changed for the ecg data we have. 

"""


# In[ ]:


"""
Extract respiration signal and respiratory rate from ECG using R-R interval.

Inspired by Sarkar et al. 2015.

---
Author: Raphael Vallat <raphaelvallat9@gmail.com>
Date: September 2018
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.interpolate import splrep, splev
from mne.filter import filter_data, resample
from scipy.signal import detrend, find_peaks
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(context='talk')

# Load and preprocess data
ecg = electrocardiogram()
sf_ori = 360
sf = 100
dsf = sf / sf_ori
ecg = resample(ecg, dsf)
ecg = filter_data(ecg, sf, 2, 30, verbose=0)

# Select only a 20 sec window
window = 20
start = 155
ecg = ecg[int(start*sf):int((start+window)*sf)]

# R-R peaks detection
rr, _ = find_peaks(ecg, distance=40, height=0.5)

plt.plot(ecg)
plt.plot(rr, ecg[rr], 'o')
plt.title('ECG signal')
plt.xlabel('Samples')
_ =plt.ylabel('Voltage')



# R-R interval in ms
rr = (rr / sf) * 1000
rri = np.diff(rr)

# Interpolate and compute HR
def interp_cubic_spline(rri, sf_up=4):
    """
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.
    
    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.
    
    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
    rri_time = np.cumsum(rri) / 1000.0
    time_rri = rri_time - rri_time[0]
    time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
    tck = splrep(time_rri, rri, s=0)
    rri_interp = splev(time_rri_interp, tck, der=0)
    return rri_interp

sf_up = 4
rri_interp = interp_cubic_spline(rri, sf_up) 
hr = 1000 * (60 / rri_interp)
print('Mean HR: %.2f bpm' % np.mean(hr))

# Detrend and normalize
edr = detrend(hr)
edr = (edr - edr.mean()) / edr.std()

# Find respiratory peaks
resp_peaks, _ = find_peaks(edr, height=0, distance=sf_up)

# Convert to seconds
resp_peaks = resp_peaks
resp_peaks_diff = np.diff(resp_peaks) / sf_up

# Plot the EDR waveform
plt.plot(edr, '-')
plt.plot(resp_peaks, edr[resp_peaks], 'o')
_ = plt.title('ECG derived respiration')

# Extract the mean respiratory rate over the selected window
mresprate = resp_peaks.size / window
print('Mean respiratory rate: %.2f Hz' % mresprate)
print('Mean respiratory period: %.2f seconds' % (1 / mresprate))
print('Respiration RMS: %.2f seconds' % np.sqrt(np.mean(resp_peaks_diff**2)))
print('Respiration STD: %.2f seconds' % np.std(resp_peaks_diff))

