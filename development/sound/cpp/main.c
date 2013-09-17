#include <stdio.h>
#include <stdlib.h>
#include <sndfile.hh>
#include <limits.h>
#include "smbPitchShift.c"
#include <cmath>
#include <iostream>
#include <algorithm>

#define THIRD_UP 1.259921
#define OCTAVE_UP 2.0
#define FIFTH_UP 1.5

using namespace std;

int main(int argc, char* argv[]) {
    SNDFILE *sf;
    SF_INFO info;
    int num_channels;
    int num, num_items;
    int *buf;
    int *shifted_pitch;
    int *summed_waves;
    float *float_buf;
    float *float_shifted_pitch;
    int f,sr,c;
    int i,j;
    FILE *out, *out2;

    /* Open the WAV file. */
    info.format = 0;
    sf = sf_open("../samples/sax.wav",SFM_READ,&info);
    if (sf == NULL)
    {
        printf("Failed to open the file.\n");
        exit(-1);
    }
    
    /* Print some of the info, and figure out how much data to read. */
    f = info.frames;
    sr = info.samplerate;
    c = info.channels;
    printf("frames=%d\n",f);
    printf("samplerate=%d\n",sr);
    printf("channels=%d\n",c);
    num_items = f*c;
    printf("num_items=%d\n",num_items);
    
    /* Allocate space for the data to be read, then read it. */
    buf = (int *) malloc(num_items*sizeof(int));
    shifted_pitch = (int *) malloc(num_items*sizeof(int));
    float_buf = (float *) malloc(num_items*sizeof(float));
    float_shifted_pitch = (float *) malloc(num_items*sizeof(float));
    summed_waves = (int *) malloc(num_items*sizeof(int));
    num = sf_read_int(sf,buf,num_items);
    sf_close(sf);
    printf("Read %d items\n",num);
    
    
    /* Write the data to filedata.out. */
    out = fopen("filedata.txt","w");
    for (i = 0; i < num; i += c)
    {
        for (j = 0; j < c; ++j) {
            float_buf[i+j] = 1.0 * buf[i+j];
            float_buf[i+j] /= INT_MAX;
            fprintf(out,"%f ",float_buf[i+j]);
        }
        fprintf(out,"\n");
    }
    fclose(out);
    
    
    smbPitchShift(1.49, num, 1024, 32, 44100, float_buf, float_shifted_pitch);
    
    float absMaxOfBuffer = 0.0;
    for (i = 0; i < num; i += c) {
        for (j = 0; j < c; ++j) {
            absMaxOfBuffer = std::max(fabs(float_shifted_pitch[i+j]), absMaxOfBuffer);
        }
    }
    
    printf("absolute max of buffer: %f\n",absMaxOfBuffer);
    
    for (i = 0; i < num; i += c)
    {
        for (j = 0; j < c; ++j) {
            //printf("old value: %f",float_shifted_pitch[i+j]);
            float_shifted_pitch[i+j] /= absMaxOfBuffer;
            //printf(", new value: %f\n",float_shifted_pitch[i+j]);            
        }
    }
    
    /* Writes normalized pitch-shifted data to filedata-shifted.out */
    out2 = fopen("filedata-shifted.txt","w");
    for (i = 0; i < num; i += c)
    {
        for (j = 0; j < c; ++j) {
            fprintf(out,"%f, ",float_shifted_pitch[i+j]);
            float_shifted_pitch[i+j] *= INT_MAX;
            shifted_pitch[i+j] = float_shifted_pitch[i+j];
            fprintf(out,"%d",shifted_pitch[i+j]);
        }
        fprintf(out2,"\n");
    }
    fclose(out2);
    
    int first_value, second_value;
    /* Sums both waves */
    for (i = 0; i < num; i += c)
    {
        for (j = 0; j < c; ++j) {
            first_value = buf[i+j];
            first_value /= 2;
            second_value = shifted_pitch[i+j];
            second_value /= 2;
            summed_waves[i+j] = first_value + second_value;
        }
    }
    fclose(out2);
    
    /* Writes a new wav file */
    
    
    const int format=SF_FORMAT_WAV | SF_FORMAT_PCM_32;
    const char* outfilename="out.wav";
        
    SndfileHandle outfile(outfilename, SFM_WRITE, format, c, sr);
    if (not outfile) return -1;
    outfile.write(summed_waves, num);
        
    return 0;

}