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

def octave_up(sound_wave, rate):
    
    high_sound_wave = []
    
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
        
    return [small_t, high_sound_wave]
    
def third_up(wave_array):

    high_sound_wave = []

    divisor = 6 # divisor > 3

    sound_wave = wave_array[1]
    rate = wave_array[2]

    for index in range(len(sound_wave)):        
        
        remainder = index % divisor
        
        average = None
        
        if remainder == 0 and len(sound_wave) > (index + 1):
            average = np.asscalar(np.int16(sound_wave[index])) + np.asscalar(np.int16(sound_wave[index + 1]))
            average /= 2
            average = np.int16(average)
            
        elif remainder == 2:
            average = np.asscalar(np.int16(sound_wave[index])) + np.asscalar(np.int16(sound_wave[index - 1]))
            average /= 2
            average = np.int16(average)
            
        elif remainder != 1 :
            average = sound_wave[index]
        
        
        if average is not None:
            high_sound_wave.append(average)
            
    lost_fraction = len(sound_wave) / divisor
    high_sound_wave += high_sound_wave[0:lost_fraction]

    #normalizes size for non-even primary waves:
    if len(high_sound_wave) > len(sound_wave):
        high_sound_wave.pop()

    high_sound_wave = high_sound_wave[:]
    high_sound_wave = np.asarray(high_sound_wave)

    small_t = np.arange(len(high_sound_wave))*1.0/rate

    return [small_t, high_sound_wave, rate]
    
def fifth_up(sound_wave, rate):

    high_sound_wave = []
    
    divisor = 3
    for index in range(len(sound_wave)):
        if index % divisor !=  1:
            high_sound_wave.append(sound_wave[index])

    
    lost_fraction = len(sound_wave) / divisor
    high_sound_wave += high_sound_wave[0:lost_fraction]

    #normalizes size for non-even primary waves:
    if len(high_sound_wave) > len(sound_wave):
        high_sound_wave.pop()

    high_sound_wave = high_sound_wave[:]
    high_sound_wave = np.asarray(high_sound_wave)

    small_t = np.arange(len(high_sound_wave))*1.0/rate

    return [high_sound_wave, small_t]
    
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
    
    print len(wave_array_1[1]), len(wave_array_2[1])
    
    wave_array_2[1] = wave_array_2[1][0:len(wave_array_1[1])]
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
            
def map_windows(wave_array):
    even_detector = 0
    window_map = [0]
    for iteration in range(len(wave_array[1])):
        last_value = np.asscalar(np.int16(wave_array[1][iteration-1]))
        current_value = np.asscalar(np.int16(wave_array[1][iteration]))
        value_multiplication = last_value * current_value
       
        if (current_value == 0 and last_value != 0) or value_multiplication < 0: 
            window_map.append(iteration)
    return window_map
    
    
def draw_window_map(window_map, wave_array):
    window_map_yaxis = []
    times_array = []
    for iteration in range(len(window_map)):
        window_map_yaxis.append(0)
        times_array.append(wave_array[0][window_map[iteration]])
    pl.plot(times_array,window_map_yaxis,'ro')

def iterative_fifth_up(wave_array, window_map):
    ## STILL TO BE DONE :)
    rate = wave_array[2]
    
    final_wave = np.copy(wave_array)
    
    final_wave[1] = final_wave[1].tolist()
    
    for iteration in range(len(window_map)/2):

        initial_index = window_map[iteration * 2]
        final_index = window_map[(iteration + 1) * 2]
        
        segment_fifth = fifth_up(wave_array[1][initial_index:final_index], rate)
       

        if iteration == 0:
            final_wave[1] = segment_fifth[0].tolist()
        else:
            final_wave[1] += segment_fifth[0].tolist()
    
    final_wave[1] = np.asarray(final_wave[1])
    
    return final_wave
    
def iterative_octave_up(wave_array, window_map):

    rate = wave_array[2]

    final_wave = np.copy(wave_array)

    final_wave[1] = final_wave[1].tolist()

    for iteration in range(len(window_map)/2):

        initial_index = window_map[iteration * 2]
        final_index = window_map[(iteration + 1) * 2]
        # if iteration == 0:
        #     segment_octave = octave_up(wave_array[1][initial_index:final_index], rate)
        # else:
        #     
        # segment_octave = octave_up(wave_array[1][initial_index:final_index], rate)
        
        #print segment_octave[0]
        segment_octave = octave_up(wave_array[1][initial_index:final_index], rate)
        
        if (iteration + 1) * 2 == len(window_map) - 2:
            break
        
        if iteration == 0:
            final_wave[1] = segment_octave[1].tolist()
            plot_wave(segment_octave)
        else:
            final_wave[1] += segment_octave[1].tolist()

    final_wave[1] = np.asarray(final_wave[1], dtype='int16')

    final_wave[0] = final_wave[0][0:len(final_wave[1])]
    # plot_wave(final_wave)
    return final_wave

np.set_printoptions(threshold='nan')
# main functions:
fil = 'Memo.wav'
wave_array = wave_to_array(fil)
#third = third_up(wave_array)
window_map_reply = map_windows(wave_array)
plot_wave(wave_array)
draw_window_map(window_map_reply, wave_array)
#pl.show()
#play_wave(third)
octave_up = iterative_fifth_up(wave_array, window_map_reply)

for iteration in range(len(octave_up[1])):
    octave_up[1][iteration] = np.int16(octave_up[1][iteration])
    
print type(wave_array[1])
print type(octave_up[1])

plot_wave(octave_up)

play_wave(wave_array)
play_wave(octave_up)
# wave_array[1] = octave_up[1]
# wave_array[0] = octave_up[0]
# wave_array[2] = octave_up[2]
# play_wave(wave_array)

summed = sum_waves(octave_up, wave_array)
play_wave(summed)
#wave_array[0] = wave_array[0][0:len(wave_array[1])]

pl.show()