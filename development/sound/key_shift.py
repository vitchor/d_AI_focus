#!/usr/bin/env python
import sys
import scipy.io.wavfile as wf
import pylab as pl
import wave
import numpy as np
import pyaudio

# wave_array format: [t, wave, rate]

def wave_to_array(wav_file):

    rate, sound_wave = wf.read(wav_file)
    
    sound_wave = sound_wave[:]
    
    big_t = np.arange(len(sound_wave))*1.0/rate
    
    return [big_t, sound_wave, rate]

def octave_up(wave_array):
    
    high_sound_wave = []
    
    sound_wave = wave_array[1]
    rate = wave_array[2]
    
    for index in range(len(sound_wave)):        
        if index % 2 != 1:
            high_sound_wave.append(sound_wave[index])
        
    high_sound_wave += high_sound_wave
    
    #normalizes size for non-even primary waves:
    if len(high_sound_wave) > len(sound_wave):
        high_sound_wave.pop()
        
    high_sound_wave = high_sound_wave[:]
    high_sound_wave = np.asarray(high_sound_wave)
        
    small_t = np.arange(len(high_sound_wave))*1.0/rate
        
    return [small_t, high_sound_wave, rate]
    
def plot_wave(wave_array):
    
    pl.plot(wave_array[0], wave_array[1])
    #pl.ylim([-100000,100000])

def play_wave(wave_array):
    
    CHUNK = 1024
    current_wave_r = wave.open(fil, 'rb')
    new_wave_w = wave.open('new_wave.wav', 'wb')
    
    p = pyaudio.PyAudio()
    
    new_wave_w.setnchannels(current_wave_r.getnchannels())
    new_wave_w.setsampwidth(current_wave_r.getsampwidth())
    new_wave_w.setframerate(current_wave_r.getframerate())
    new_wave_w.setnframes(current_wave_r.getnframes())
    #new_wave_w.setcomptype(current_wave_r.getcomptype())
    
    new_wave_w.writeframesraw(wave_array[1])
    
    new_wave_w.close()
    
    #new_wave_r = wave.open('440Hz.wav', 'rb')
    new_wave_r = wave.open('new_wave.wav', 'rb')
    
    stream = p.open(format=p.get_format_from_width(new_wave_r.getsampwidth()),
                    channels=new_wave_r.getnchannels(),
                    rate=new_wave_r.getframerate(),
                    output=True)
    
    data = new_wave_r.readframes(CHUNK)
    
    while data != '':
        stream.write(data)
        data = new_wave_r.readframes(CHUNK)
    
    stream.stop_stream()
    stream.close()
    #new_wave_r.close()
    
    p.terminate()
    
def sum_waves(wave_array_1, wave_array_2):
    if len(wave_array_1[1]) == len(wave_array_2[1]):
        result_wave = []
        for index in range(len(wave_array_1[1])):
            wave_array_1[1][index] = wave_array_1[1][index]/2
            wave_array_2[1][index] = wave_array_2[1][index]/2
            result = wave_array_1[1][index] + wave_array_2[1][index]
            result_wave.append(result)
        result_wave = result_wave[:]
        result_wave = np.asarray(result_wave)
        
        return [wave_array_1[0], result_wave, wave_array_1[2]]
    else:
        return 0
            
# main functions:
fil = '1Hz.wav'

wave_array = wave_to_array(fil)
play_wave(wave_array)
first_octave_array = octave_up(wave_array)
second_octave_array = octave_up(first_octave_array)
#play_wave(first_octave_array)
summed_wave = sum_waves(wave_array, first_octave_array)
summed_wave = sum_waves(summed_wave, second_octave_array)
plot_wave(summed_wave)
pl.show()
#plot_wave(first_octave_array)
play_wave(summed_wave)
#pl.show()
#print len(summed_wave[0])



#play_wave(first_octave_array)
#play_wave(second_octave_array)