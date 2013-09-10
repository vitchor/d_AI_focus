import scipy
import wave
import struct
import numpy
import pylab
import sys

fp = wave.open(sys.argv[1], 'rb')

samplerate = fp.getframerate()
totalsamples = fp.getnframes()
fft_length = 256 # Guess
num_fft = (totalsamples / fft_length) - 2

#print (samplerate)

temp = numpy.zeros((num_fft, fft_length), float)

for i in range(num_fft):

    tempb = fp.readframes(fft_length / fp.getnchannels() / fp.getsampwidth());

    up = (struct.unpack("%dB"%(fft_length), tempb))

    temp[i,:] = numpy.array(up, float) - 128.0

temp = temp * numpy.hamming(fft_length)

temp.shape = (-1, fp.getnchannels())

fftd = numpy.fft.fft(temp)

pylab.plot(abs(fftd[:,1]))

pylab.show()