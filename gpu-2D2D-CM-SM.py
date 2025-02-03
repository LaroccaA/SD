import os
import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# CUDA kernel con l'uso della constant memory per i filtri Sobel
cuda_code = """
__constant__ float horizontal_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__constant__ float vertical_filter[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

__device__ inline unsigned char clamp(int value) {
    return (value < 0) ? 0 : ((value > 255) ? 255 : value);
}

__global__ void sobelFilterKernel(unsigned char* d_input, unsigned char* d_output, int width, int height, int filterSize) {
    extern __shared__ unsigned char shared_mem[];
    int radius = filterSize / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_x = threadIdx.x + radius;
    int local_y = threadIdx.y + radius;

    // Load data into shared memory
    if (x < width && y < height) {
        shared_mem[local_y * (blockDim.x + 2 * radius) + local_x] = d_input[y * width + x];

        // Handle borders in shared memory
        if (threadIdx.x < radius) {
            if (x >= radius) shared_mem[local_y * (blockDim.x + 2 * radius) + local_x - radius] = d_input[y * width + (x - radius)];
            if (x + blockDim.x < width) shared_mem[local_y * (blockDim.x + 2 * radius) + local_x + blockDim.x] = d_input[y * width + (x + blockDim.x)];
        }
        if (threadIdx.y < radius) {
            if (y >= radius) shared_mem[(local_y - radius) * (blockDim.x + 2 * radius) + local_x] = d_input[(y - radius) * width + x];
            if (y + blockDim.y < height) shared_mem[(local_y + blockDim.y) * (blockDim.x + 2 * radius) + local_x] = d_input[(y + blockDim.y) * width + x];
        }
    }

    __syncthreads();

    // Apply Sobel filter
    if (x < width && y < height) {
        float horizontal_result = 0.0f;
        float vertical_result = 0.0f;

        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                int shared_index = (local_y + fy - radius) * (blockDim.x + 2 * radius) + (local_x + fx - radius);
                unsigned char pixel_value = shared_mem[shared_index];
                horizontal_result += pixel_value * horizontal_filter[fy][fx];
                vertical_result += pixel_value * vertical_filter[fy][fx];
            }
        }

        float magnitude = sqrtf(horizontal_result * horizontal_result + vertical_result * vertical_result);
        d_output[y * width + x] = clamp((int)magnitude);
    }
}
"""

# Compile the CUDA code
mod = SourceModule(cuda_code)
sobel_filter_kernel = mod.get_function("sobelFilterKernel")

def process_single_image(image_path, output_folder):
    # Load the image
    image = Image.open(image_path).convert('L')
    image_np = np.array(image, dtype=np.uint8)
    height, width = image_np.shape

    # Allocate memory for input and output
    d_input = cuda.mem_alloc(image_np.nbytes)
    d_output = cuda.mem_alloc(image_np.nbytes)
    cuda.memcpy_htod(d_input, image_np)

    # Test different block sizes
    block_sizes = [(16, 16)]
    for block_size_x, block_size_y in block_sizes:
        block = (block_size_x, block_size_y, 1)
        grid = ((width + block_size_x - 1) // block_size_x, (height + block_size_y - 1) // block_size_y, 1)
        shared_mem_size = (block_size_x + 2) * (block_size_y + 2)

        print(f"  Running kernel with block size ({block_size_x}, {block_size_y})...")

        # Launch the kernel
        start_time = time.time()
        sobel_filter_kernel(
            d_input,
            d_output,
            np.int32(width),
            np.int32(height),
            np.int32(3),  # Sobel filter size
            block=block,
            grid=grid,
            shared=shared_mem_size
        )
        cuda.Context.synchronize()
        elapsed_time = time.time() - start_time
        print(f"    Completed in {elapsed_time:.4f} seconds.")

        # Copy the result back to host
        output_np = np.empty_like(image_np)
        cuda.memcpy_dtoh(output_np, d_output)

        # Save the output image for this configuration
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_name = f"{image_name}_block_{block_size_x}x{block_size_y}_shared.png"
        output_image_path = os.path.join(output_folder, output_image_name)
        output_image = Image.fromarray(output_np)
        output_image.save(output_image_path)
        print(f"    Output saved as {output_image_path}")

if __name__ == "__main__":
    input_folder = "Dataset"  # Path to input folder
    output_folder = "Output"  # Path to output folder

    os.makedirs(output_folder, exist_ok=True)

    # Process each image one at a time
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            print(f"Processing image: {image_path}")
            process_single_image(image_path, output_folder)
