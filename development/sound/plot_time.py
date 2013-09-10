'''
import scipy.io.wavfile as wavfile
import numpy as np
import pylab as pl
import sys

rate, data = wavfile.read(sys.argv[1])
t = np.arange(len(data[:,0]))*1.0/rate
pl.plot(t, data[:,0])
pl.show()

'''
#!/usr/bin/env python
import sys
import scipy.io.wavfile as wf
import pylab as pl
import wave
import numpy as np

def showWaveAndSpec(wavFile):

    rate, soundWave = wf.read(wavFile)
    t = np.arange(len(soundWave[:,0]))*1.0/rate
    pl.plot(soundWave[:,0])
    pl.show()

    '''
    f = spf.getframerate()
    
    soundInfo = wavfile.read
    
    subplot(211)
    plot(sound_info)
    title('Wave from and spectrogram of %s' % sys.argv[1])

    subplot(212)
    spectrogram = specgram(sound_info, Fs = f, scale_by_freq=True,sides='default')
    
    show()
    spf.close()
    '''

fil = sys.argv[1]

showWaveAndSpec(fil)
