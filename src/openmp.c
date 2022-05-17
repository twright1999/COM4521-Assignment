#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>

///
/// Algorithm storage
///
Image omp_input_image;
Image omp_output_image;
unsigned int TILES_X, TILES_Y;
unsigned long long* mosaic_sum;
unsigned char* mosaic_value;

unsigned char* compact_mosaic;
unsigned char* global_pixel_average;

void openmp_begin(const Image *input_image) {
    TILES_X = input_image->width / TILE_SIZE;
    TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    mosaic_sum = (unsigned long long*)malloc(TILES_X * TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    mosaic_value = (unsigned char*)malloc(TILES_X * TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    omp_input_image = *input_image;
    omp_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(omp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    omp_output_image = *input_image;
    omp_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
}
void openmp_stage1() {
    // Reset sum memory to 0
    memset(mosaic_sum, 0, TILES_X * TILES_Y * omp_input_image.channels * sizeof(unsigned long long));

    // def iterators for pragma loop
    int t_x, t_y, p_x, p_y, ch;
    // Sum pixel data within each tile
#pragma omp parallel for private(t_x, t_y, p_x, p_y, ch)
    for (t_x = 0; t_x < TILES_X; ++t_x) {
        for (t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * TILES_X + t_x) * omp_input_image.channels;
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;
            for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    // For each colour channel
                    const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;
                    for (ch = 0; ch < omp_input_image.channels; ++ch) {
                        // Load pixel
                        const unsigned char pixel = omp_input_image.data[tile_offset + pixel_offset + ch];
                        mosaic_sum[tile_index + ch] += pixel;
                    }
                }
            }
        }
    }

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(&omp_input_image, mosaic_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&omp_input_image, mosaic_sum);
#endif
}
void openmp_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    for (unsigned int t = 0; t < TILES_X * TILES_Y; ++t) {
        for (int ch = 0; ch < omp_input_image.channels; ++ch) {
            mosaic_value[t * omp_input_image.channels + ch] = (unsigned char)(mosaic_sum[t * omp_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
            whole_image_sum[ch] += mosaic_value[t * omp_input_image.channels + ch];
        }
    }
    // Reduce the whole image sum to whole image average for the return value
    for (int ch = 0; ch < omp_input_image.channels; ++ch) {
        output_global_average[ch] = (unsigned char)(whole_image_sum[ch] / (TILES_X * TILES_Y));
    }

    // assign input values to output values to skip stage
    // compact_mosaic = mosaic_value;
    // global_pixel_average = output_global_average;

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    validate_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}
void openmp_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    // For each tile
    for (unsigned int t_x = 0; t_x < TILES_X; ++t_x) {
        for (unsigned int t_y = 0; t_y < TILES_Y; ++t_y) {
            const unsigned int tile_index = (t_y * TILES_X + t_x) * omp_input_image.channels;
            const unsigned int tile_offset = (t_y * TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;

            // For each pixel within the tile
            for (unsigned int p_x = 0; p_x < TILE_SIZE; ++p_x) {
                for (unsigned int p_y = 0; p_y < TILE_SIZE; ++p_y) {
                    const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;
                    // Copy whole pixel
                    memcpy(omp_output_image.data + tile_offset + pixel_offset, mosaic_value + tile_index, omp_input_image.channels);
                }
            }
        }
    }
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(&omp_input_image, mosaic_value, &omp_output_image);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_broadcast(&omp_input_image, mosaic_value, &omp_output_image);
#endif    
}
void openmp_end(Image *output_image) {
    
}