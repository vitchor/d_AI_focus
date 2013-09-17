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

// GLOBAL VARIABLES:
int i_num_channels, i_wave_length, i_wave_num_of_itens;
int i_wave_frames, i_wave_sample_rate, i_wave_channels;

void get_info(char* c_in_wav_file_name) {
    SNDFILE *m_wave_file;
    SF_INFO m_wave_info;
    
    // Opens wav file:
    m_wave_info.format = 0;
    m_wave_file = sf_open(c_in_wav_file_name,SFM_READ,&m_wave_info);
    if (not m_wave_file) {
        printf("Failed to open WAV file to read.\n");
        return;
    }
    
    // Gathers information from wav file:
    i_wave_frames = m_wave_info.frames;
    i_wave_sample_rate = m_wave_info.samplerate;
    i_wave_channels = m_wave_info.channels;
    i_wave_num_of_itens = i_wave_frames*i_wave_channels;
    
    sf_close(m_wave_file);
}

void wave_to_array(char* c_in_wav_file_name, int* i_out_wave_array) {
    //printf("File name: %s", c_in_wav_file_name);
    
    SNDFILE *m_wave_file;
    SF_INFO m_wave_info;
    
    // Opens wave file
    m_wave_info.format = 0;
    m_wave_file = sf_open(c_in_wav_file_name,SFM_READ,&m_wave_info);
    if (not m_wave_file) {
        printf("Failed to open WAV file to read.\n");
        return;
    }
    
    // Writes on integer range the wave to i_out_wave_array
    i_wave_length = sf_read_int(m_wave_file,i_out_wave_array,i_wave_num_of_itens);
    sf_close(m_wave_file);
}

void pitch_shift(int* i_in_wave_array, int* i_out_wave_array, float f_ratio_shift) {
    float *f_wave_array;
    float *f_wave_shifted_pitch;
    float f_byte_value;
    
    f_wave_array = (float *) malloc(i_wave_num_of_itens*sizeof(float));
    f_wave_shifted_pitch = (float *) malloc(i_wave_num_of_itens*sizeof(float));

    //int *i_wave_array;
    //int *i_wave_shifted_pitch;
    
    // Normalizes the wave data from wave_array to the interval [-1,1]
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_wave_array[i] = 1.0 * i_in_wave_array[i];
        f_wave_array[i] /= INT_MAX;
    }
    
    smbPitchShift(f_ratio_shift, i_wave_length, 1024, 32, 44100, f_wave_array, f_wave_shifted_pitch);
    
    // Gets max absolute value from f_wave_shifted_pitch so as to normalize it back to [-1,1]
    float f_abs_max_of_wave = 0.0;
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_abs_max_of_wave = std::max(fabs(f_wave_shifted_pitch[i]), f_abs_max_of_wave);
    }
    
    // Normalizes f_wave_shifted_pitch back to [-1,1]. Converts the result back as an integer on i_wave_shifted_pitch
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        f_wave_shifted_pitch[i] /= f_abs_max_of_wave;
        f_byte_value = f_wave_shifted_pitch[i] * INT_MAX;
        i_out_wave_array[i] = f_byte_value;
    }
}

void array_to_wave(char* c_out_wav_file_name, int* i_in_wave_array) {
    // Writes a wav file with the shifted pitch
    const int i_format=SF_FORMAT_WAV | SF_FORMAT_PCM_32;
    
    SndfileHandle h_outfile(c_out_wav_file_name, SFM_WRITE, i_format, i_wave_channels, i_wave_sample_rate);
    if (not h_outfile) {
        printf("Failed to open WAV file to write.\n");
        return;
    }
    h_outfile.write(i_in_wave_array, i_wave_length);
}

int main(int argc, char* argv[]) {
    // argv[1] = origin_wav_file_name
    
    char* c_in_file = argv[1];
    
    int *i_original_wave_array;
    int *i_third_up_wave_array;
    int *i_fifth_up_wave_array;
    int *i_summed_triads_array;
    
    // Saves info from wave on global vars
    get_info(c_in_file);
    
    // Allocating space for all wave arrays needed in this function
    i_original_wave_array = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    i_third_up_wave_array = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    i_fifth_up_wave_array = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    i_summed_triads_array = (int *) malloc(i_wave_num_of_itens*sizeof(int));
    
    // Generates wave arrays
    wave_to_array(c_in_file, i_original_wave_array);
    pitch_shift(i_original_wave_array, i_third_up_wave_array, THIRD_UP);
    pitch_shift(i_original_wave_array, i_fifth_up_wave_array, FIFTH_UP);    
    
    // Sums all three waves
    int i_first_value, i_second_value, i_third_value;
    for (int i = 0; i < i_wave_length; i += i_wave_channels) {
        i_first_value = i_original_wave_array[i];
        i_first_value /= 3;
        i_second_value = i_third_up_wave_array[i];
        i_second_value /= 3;
        i_third_value = i_fifth_up_wave_array[i];
        i_third_value /= 3;
        i_summed_triads_array[i] = i_first_value + i_second_value + i_third_value;
    }
    
    // Writes the wav file for the triads
    char* c_out_file_name = "output/triad.wav";
    array_to_wave(c_out_file_name, i_summed_triads_array);
    
    return 0;
}