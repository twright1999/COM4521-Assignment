#include "cuda.cuh"
#include <cstring>
#include "helper.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define THREADS_PER_BLOCK TILE_SIZE * TILE_SIZE

///
/// Algorithm storage
///
// Host copy of input image
Image cuda_input_image;
// Host copy of image tiles in each dimension
unsigned int cuda_TILES_X, cuda_TILES_Y;
// Pointer to device buffer for calculating the sum of each tile mosaic, this must be passed to a kernel to be used on device
unsigned long long* d_mosaic_sum;
// Pointer to device buffer for storing the output pixels of each tile, this must be passed to a kernel to be used on device
unsigned char* d_mosaic_value;
// Pointer to device image data buffer, for storing the input image, this must be passed to a kernel to be used on device
unsigned char* d_input_image_data;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;
// Pointer to device buffer for the global pixel average sum, this must be passed to a kernel to be used on device
unsigned long long* d_global_pixel_sum;

// Host copy of mosaic sum for validation
unsigned long long* cuda_mosaic_sum;
// Host copy of mosaic value for validation
unsigned char* cuda_mosaic_value;
// Host copy of output image for validation
Image cuda_output_image;

// device variables
__device__ int d_input_image_width;
__device__ int d_input_image_channels;
__device__ unsigned int d_TILES_X;

void cuda_begin(const Image *input_image) {
    // These are suggested CUDA memory allocations that match the CPU implementation
    // If you would prefer, you can rewrite this function (and cuda_end()) to suit your preference

    cuda_TILES_X = input_image->width / TILE_SIZE;
    cuda_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    CUDA_CALL(cudaMalloc(&d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned long long)));

    // Allocate buffer for storing the output pixel value of each tile
    CUDA_CALL(cudaMalloc(&d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * input_image->channels * sizeof(unsigned char)));

    const size_t image_data_size = input_image->width * input_image->height * input_image->channels * sizeof(unsigned char);
    // Allocate copy of input image
    cuda_input_image = *input_image;
    cuda_input_image.data = (unsigned char*)malloc(image_data_size);
    memcpy(cuda_input_image.data, input_image->data, image_data_size);

    // Allocate and fill device buffer for storing input image data
    CUDA_CALL(cudaMalloc(&d_input_image_data, image_data_size));
    CUDA_CALL(cudaMemcpy(d_input_image_data, input_image->data, image_data_size, cudaMemcpyHostToDevice));

    // Allocate device buffer for storing output image data
    CUDA_CALL(cudaMalloc(&d_output_image_data, image_data_size));

    // Allocate and zero buffer for calculation global pixel average
    CUDA_CALL(cudaMalloc(&d_global_pixel_sum, input_image->channels * sizeof(unsigned long long)));

    // Copy host variables to the host
    CUDA_CALL(cudaMemcpyToSymbol(d_input_image_width, &input_image->width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(d_input_image_channels, &input_image->channels, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(d_TILES_X, &cuda_TILES_X, sizeof(unsigned int)));

    // Allocate host mosaic sum for validation
    cuda_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    // Allocate host mosaic value for validation
    cuda_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char));
    // Allocate host output image for validation
    cuda_output_image.data = (unsigned char*)malloc(image_data_size);
}

__global__ void stage1(unsigned char *d_input_image_data, unsigned long long* d_mosaic_sum) {

    unsigned int t_x = blockIdx.x;
    unsigned int t_y = blockIdx.y;
    unsigned int p_x = threadIdx.x;
    unsigned int p_y = threadIdx.y;

    unsigned int ch = blockIdx.z;

    const unsigned int tile_index = (t_y * d_TILES_X + t_x) * d_input_image_channels;
    const unsigned int tile_offset = (t_y * d_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * d_input_image_channels;
    const unsigned int pixel_offset = (p_y * d_input_image_width + p_x) * d_input_image_channels;


    const unsigned char pixel = d_input_image_data[tile_offset + pixel_offset + ch];
    atomicAdd(&d_mosaic_sum[tile_index + ch], pixel);
}

void cuda_stage1() {
    unsigned int block_width = TILE_SIZE;
    unsigned int grid_width = (unsigned int)ceil((double)cuda_input_image.width / block_width);
    unsigned int grid_height = (unsigned int)ceil((double)cuda_input_image.height / block_width);

    dim3 blocksPerGrid(grid_width, grid_height, 3);
    dim3 threadsPerBlock(block_width, block_width, 1);

    stage1 << <blocksPerGrid, threadsPerBlock >> > (d_input_image_data, d_mosaic_sum);

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(&cuda_input_image, d_mosaic_sum);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    validate_tile_sum(&cuda_input_image, cuda_mosaic_sum);
#endif
}
void cuda_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    //unsigned long long whole_image_sum[4] = { 0, 0, 0, 0 };  // Only 3 is required for the assignment, but this version hypothetically supports upto 4 channels
    //for (unsigned int t = 0; t < cuda_TILES_X * cuda_TILES_Y; ++t) {
    //    for (int ch = 0; ch < cuda_input_image.channels; ++ch) {
    //        d_mosaic_value[t * cuda_input_image.channels + ch] = (unsigned char)(d_mosaic_sum[t * cuda_input_image.channels + ch] / TILE_PIXELS);  // Integer division is fine here
    //        whole_image_sum[ch] += d_mosaic_value[t * cuda_input_image.channels + ch];
    //    }
    //}

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, d_mosaic_sum, d_mosaic_value, output_global_average);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    // validate_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, mosaic_value, output_global_average);
#endif    
}

__global__ void stage3(unsigned char* d_input_image_data, unsigned char* d_output_image_data, unsigned char* d_mosaic_value) {
    unsigned int t_x = blockIdx.x;
    unsigned int t_y = blockIdx.y;
    unsigned int p_x = threadIdx.x;
    unsigned int p_y = threadIdx.y;

    const unsigned int tile_index = (t_y * d_TILES_X + t_x) * d_input_image_channels;
    const unsigned int tile_offset = (t_y * d_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * d_input_image_channels;
    const unsigned int pixel_offset = (p_y * d_input_image_width + p_x) * d_input_image_channels;


    // Time ~0.546ms (4096x4096)
    //memcpy(d_output_image_data + tile_offset + pixel_offset, d_mosaic_value + tile_index, d_input_image_channels);


    // Time ~0.469ms (4096x4096)
    // unsigned int ch = blockIdx.z;
    // d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[tile_index + ch];


    // Best time ~0.54ms (4096x4096)
    d_output_image_data[tile_offset + pixel_offset] = d_mosaic_value[tile_index];
    d_output_image_data[tile_offset + pixel_offset + 1] = d_mosaic_value[tile_index + 1];
    d_output_image_data[tile_offset + pixel_offset + 2] = d_mosaic_value[tile_index + 2];
}
void cuda_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    
    unsigned int block_width = TILE_SIZE;
    unsigned int grid_width = (unsigned int)ceil((double)cuda_input_image.width / block_width);
    unsigned int grid_height = (unsigned int)ceil((double)cuda_input_image.height / block_width);

    dim3 blocksPerGrid(grid_width, grid_height, 1);
    dim3 threadsPerBlock(block_width, block_width, 1);

    stage3 << <blocksPerGrid, threadsPerBlock >> > (d_input_image_data, d_output_image_data, d_mosaic_value);

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_broadcast(&cuda_input_image, d_mosaic_value, &cuda_input_image);

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaMemcpy(cuda_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    const size_t image_data_size = cuda_input_image.width * cuda_input_image.height * cuda_input_image.channels * sizeof(unsigned char);
    cudaMemcpy(cuda_output_image.data, d_output_image_data, image_data_size, cudaMemcpyDeviceToHost);
    validate_broadcast(&cuda_input_image, cuda_mosaic_value, &cuda_output_image);
#endif    
}
void cuda_end(Image *output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    output_image->width = cuda_input_image.width;
    output_image->height = cuda_input_image.height;
    output_image->channels = cuda_input_image.channels;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    free(cuda_input_image.data);
    CUDA_CALL(cudaFree(d_mosaic_value));
    CUDA_CALL(cudaFree(d_mosaic_sum));
    CUDA_CALL(cudaFree(d_input_image_data));
    CUDA_CALL(cudaFree(d_output_image_data));

    // Added release allocations
    free(cuda_mosaic_sum);
    free(cuda_mosaic_value);
    free(cuda_output_image.data);
}
