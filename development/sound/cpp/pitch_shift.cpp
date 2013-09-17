// ------------------------------------------------------------------------------
// Compile me with:
//  $ g++ pitch_shift.cpp -o pitch_shift -lsndfile
//
// NOTE: to install libsndfile on mac, use:
//  $ sudo port install libsndfile
// ------------------------------------------------------------------------------
// CODE START

// Basic C/C++ libs needed for this program
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

// Additional libs required
#include "lib/smbPitchShift.c"
#include <sndfile.hh>

// Music frequencies definitions
#define THIRD_UP 1.259921
#define OCTAVE_UP 2.0
#define FIFTH_UP 1.5

using namespace std;

// Method for shifting a .WAV file's pitch in a given frequency ration, writing the output on .wav files
int main(int argc, char** argv) {
    // argv[1] = name
    // argv[2] = ratio
    float f_ratio = FIFTH_UP;
    char* c_in_file = "../samples/sax.wav";
    //char* c_in_file = "summed.wav";
    const char* c_outfilename = "shifted_frequency.wav";
    const char* c_summedfilename="summed.wav";

    SNDFILE *m_wave_file;
    SF_INFO m_wave_info;
    int i_num_channels, i_wave_length, i_wave_num_of_itens;
    int i_wave_frames, i_wave_sample_rate, i_wave_channels;
    int *i_wave_array;
    int *i_wave_shifted_pitch;
    int *i_summed_waves;
    
    float *f_wave_array;
    float *f_wave_shifted_pitch;
    
    // Opens wave file
    m_wave_info.format = 0;
    m_wave_file = sf_open(c_in_file,SFM_READ,&m_wave_info);
    if (not m_wave_file) {
        printf("Failed to open WAV file to read.\n");
        return -1;
    }
    
    // Gathers information from wav file:
    i_wave_frames = m_wave_info.frames;
    i_wave_sample_rate = m_wave_info.samplerate;
    i_wave_channels = m_wave_info.channels;
    i_wave_num_of_itens = i_wave_frames*i_wave_channels;
    
    // Allocating space for all wave arrays needed in this function
    i_wave_array = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    i_wave_shifted_pitch = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    i_summed_waves = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    f_wave_array = (float *) malloc(i_wave_num_of_itens*sizeof(float));
    f_wave_shifted_pitch = (float *) malloc(i_wave_num_of_itens*sizeof(float));
    i_wave_length = sf_read_int(m_wave_file,i_wave_array,i_wave_num_of_itens);
    sf_close(m_wave_file);
    
    // Normalizes the wave data from wave_array to the interval [-1,1]
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_wave_array[i] = 1.0 * i_wave_array[i];
        f_wave_array[i] /= INT_MAX;
    }
    
    smbPitchShift(f_ratio, i_wave_length, 1024, 32, 44100, f_wave_array, f_wave_shifted_pitch);
    
    // Gets max absolute value from f_wave_shifted_pitch so as to normalize it back to [-1,1]
    float f_abs_max_of_wave = 0.0;
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_abs_max_of_wave = std::max(fabs(f_wave_shifted_pitch[i]), f_abs_max_of_wave);
    }
    
    // Normalizes f_wave_shifted_pitch back to [-1,1]. Converts the result back as an integer on i_wave_shifted_pitch
    float f_byte_value;
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_wave_shifted_pitch[i] /= f_abs_max_of_wave;
        f_byte_value = f_wave_shifted_pitch[i] * INT_MAX;
        i_wave_shifted_pitch[i] = f_byte_value;
    }
    
    // Writes a wav file with the shifted pitch
    const int i_format=SF_FORMAT_WAV | SF_FORMAT_PCM_32;
    
    SndfileHandle h_outfile(c_outfilename, SFM_WRITE, i_format, i_wave_channels, i_wave_sample_rate);
    if (not h_outfile) {
        printf("Failed to open WAV file to write.\n");
        return -1;
    }
    h_outfile.write(i_wave_shifted_pitch, i_wave_length);
    
    // Sums both waves
    int i_first_value, i_second_value;
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        i_first_value = i_wave_array[i];
        i_first_value /= 2;
        i_second_value = i_wave_shifted_pitch[i];
        i_second_value /= 2;
        i_summed_waves[i] = i_first_value + i_second_value;
    }
    
    // Writes a wav file with the sum of the original + shifted pitches
    SndfileHandle h_outfilesummed(c_summedfilename, SFM_WRITE, i_format, i_wave_channels, i_wave_sample_rate);
    if (not h_outfilesummed) {
        printf("Failed to open WAV file to write.\n");
        return -1;
    }
    h_outfilesummed.write(i_summed_waves, i_wave_length);
    
    return 0;
}