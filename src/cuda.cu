#include "cuda.cuh"
#include <cstring>
#include "helper.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

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
// Host copy of global pixel sum
unsigned long long* cuda_global_pixel_sum;


// Device copy of output global average
//unsigned char* d_output_global_average;

// device variables
__device__ int d_input_image_width;
__device__ int d_input_image_channels;
__device__ unsigned int d_TILES_X;
__device__ unsigned int d_TILES_Y;

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
    CUDA_CALL(cudaMemcpyToSymbol(d_TILES_Y, &cuda_TILES_Y, sizeof(unsigned int)));

    // Allocate host mosaic sum for validation
    cuda_mosaic_sum = (unsigned long long*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long));
    // Allocate host mosaic value for validation
    cuda_mosaic_value = (unsigned char*)malloc(cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char));
    // Allocate host output image for validation
    cuda_output_image.data = (unsigned char*)malloc(image_data_size);
    // Allocate host global pixel sum for validation
    cuda_global_pixel_sum = (unsigned long long*)malloc(cuda_input_image.channels * sizeof(unsigned long long));

    // Allocate device for copy of output global average
    //CUDA_CALL(cudaMalloc(&d_output_global_average, cuda_input_image.channels * sizeof(unsigned char)));
}

__global__ void stage1(unsigned char *d_input_image_data, unsigned long long* d_mosaic_sum) {
    unsigned int t_x = blockIdx.x;
    unsigned int t_y = blockIdx.y;
    unsigned int p_x = threadIdx.x;
    unsigned int p_y = threadIdx.y;

    const unsigned int tile_index = (t_y * d_TILES_X + t_x) * d_input_image_channels;
    const unsigned int tile_offset = (t_y * d_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * d_input_image_channels;
    const unsigned int pixel_offset = (p_y * d_input_image_width + p_x) * d_input_image_channels;

    unsigned int pixel_r_sum = d_input_image_data[tile_offset + pixel_offset];
    unsigned int pixel_g_sum = d_input_image_data[tile_offset + pixel_offset + 1];
    unsigned int pixel_b_sum = d_input_image_data[tile_offset + pixel_offset + 2];

    for (int offset = 16; offset > 0; offset /= 2) {
        pixel_r_sum += __shfl_down(pixel_r_sum, offset);
        pixel_g_sum += __shfl_down(pixel_g_sum, offset);
        pixel_b_sum += __shfl_down(pixel_b_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&d_mosaic_sum[tile_index], pixel_r_sum);
        atomicAdd(&d_mosaic_sum[tile_index + 1], pixel_g_sum);
        atomicAdd(&d_mosaic_sum[tile_index + 2], pixel_b_sum);
    }
}

void cuda_stage1() {
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);

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

__global__ void stage2(unsigned long long * d_mosaic_sum, unsigned char* d_mosaic_value, unsigned long long* d_global_pixel_sum) {
    //, unsigned char* d_output_global_average
    unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;

    d_mosaic_value[t * d_input_image_channels] = (unsigned char)(d_mosaic_sum[t * d_input_image_channels] / TILE_PIXELS);
    d_mosaic_value[t * d_input_image_channels + 1] = (unsigned char)(d_mosaic_sum[t * d_input_image_channels + 1] / TILE_PIXELS);
    d_mosaic_value[t * d_input_image_channels + 2] = (unsigned char)(d_mosaic_sum[t * d_input_image_channels + 2] / TILE_PIXELS);

    unsigned int r_sum = d_mosaic_value[t * d_input_image_channels];
    unsigned int g_sum = d_mosaic_value[t * d_input_image_channels + 1];
    unsigned int b_sum = d_mosaic_value[t * d_input_image_channels + 2];

    for (int offset = 16; offset > 0; offset /= 2) {
        r_sum += __shfl_down(r_sum, offset);
        g_sum += __shfl_down(g_sum, offset);
        b_sum += __shfl_down(b_sum, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&d_global_pixel_sum[0], r_sum);
        atomicAdd(&d_global_pixel_sum[1], g_sum);
        atomicAdd(&d_global_pixel_sum[2], b_sum);
    }
}
void cuda_stage2(unsigned char* output_global_average) {
    unsigned int total_tiles = cuda_TILES_X * cuda_TILES_Y;
    unsigned int grid_size = (unsigned int)ceil(((double)(cuda_TILES_X * cuda_TILES_Y)) / 1024);
    unsigned int threads_per_block = (unsigned int)ceil((double)total_tiles / grid_size);

    dim3 blocksPerGrid(grid_size, 1, 1);
    dim3 threadsPerBlock(threads_per_block, 1, 1);

    stage2 << <blocksPerGrid, threadsPerBlock >> > (d_mosaic_sum, d_mosaic_value, d_global_pixel_sum);

    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, d_mosaic_sum, d_mosaic_value, output_global_average);

    cudaMemcpy(cuda_global_pixel_sum, d_global_pixel_sum, cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    output_global_average[0] = (unsigned char)(cuda_global_pixel_sum[0] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[1] = (unsigned char)(cuda_global_pixel_sum[1] / (cuda_TILES_X * cuda_TILES_Y));
    output_global_average[2] = (unsigned char)(cuda_global_pixel_sum[2] / (cuda_TILES_X * cuda_TILES_Y));

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaMemcpy(cuda_mosaic_sum, d_mosaic_sum, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_mosaic_value, d_mosaic_value, cuda_TILES_X * cuda_TILES_Y * cuda_input_image.channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    validate_compact_mosaic(cuda_TILES_X, cuda_TILES_Y, cuda_mosaic_sum, cuda_mosaic_value, output_global_average);
#endif    
}

__global__ void stage3(unsigned char* d_output_image_data, unsigned char* d_mosaic_value) {
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
    //unsigned int ch = blockIdx.z;
    //d_output_image_data[tile_offset + pixel_offset + ch] = d_mosaic_value[tile_index + ch];


    // Best time ~0.54ms (4096x4096)
    d_output_image_data[tile_offset + pixel_offset] = d_mosaic_value[tile_index];
    d_output_image_data[tile_offset + pixel_offset + 1] = d_mosaic_value[tile_index + 1];
    d_output_image_data[tile_offset + pixel_offset + 2] = d_mosaic_value[tile_index + 2];
}
void cuda_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    
    dim3 blocksPerGrid(cuda_TILES_X, cuda_TILES_Y, 1);
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE, 1);

    stage3 << <blocksPerGrid, threadsPerBlock >> > (d_output_image_data, d_mosaic_value);

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
    CUDA_CALL(cudaFree(d_global_pixel_sum));

    // Added release allocations
    free(cuda_mosaic_sum);
    free(cuda_mosaic_value);
    free(cuda_output_image.data);
    free(cuda_global_pixel_sum);
    //CUDA_CALL(cudaFree(d_output_global_average));
}
