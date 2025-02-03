import os
import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# CUDA kernel ottimizzato
cuda_code = """
__constant__ float horizontal_filter[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
__constant__ float vertical_filter[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

__device__ inline unsigned char clamp(int value) {
    return (value < 0) ? 0 : ((value > 255) ? 255 : value);
}

__global__ void sobelFilterKernel(unsigned char* d_input, unsigned char* d_output, int width, int height) {
    extern __shared__ unsigned char shared_mem[];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int local_x = threadIdx.x + 1; // Offset per il raggio (1 per Sobel 3x3)
    int local_y = threadIdx.y + 1;

    int shared_stride = blockDim.x + 2; // Include i bordi per il padding

    // Carica la memoria condivisa (con gestione del bordo)
    if (x < width && y < height) {
        shared_mem[local_y * shared_stride + local_x] = d_input[y * width + x];

        // Carica i bordi della shared memory
        if (threadIdx.x == 0 && x > 0) {
            shared_mem[local_y * shared_stride + local_x - 1] = d_input[y * width + x - 1];
        }
        if (threadIdx.x == blockDim.x - 1 && x < width - 1) {
            shared_mem[local_y * shared_stride + local_x + 1] = d_input[y * width + x + 1];
        }
        if (threadIdx.y == 0 && y > 0) {
            shared_mem[(local_y - 1) * shared_stride + local_x] = d_input[(y - 1) * width + x];
        }
        if (threadIdx.y == blockDim.y - 1 && y < height - 1) {
            shared_mem[(local_y + 1) * shared_stride + local_x] = d_input[(y + 1) * width + x];
        }
    }

    __syncthreads();

    // Applica il filtro Sobel solo se all'interno dei limiti dell'immagine
    if (x < width && y < height) {
        float horizontal_result = 0.0f;
        float vertical_result = 0.0f;

        // Calcolo diretto senza loop annidati
        horizontal_result =
            shared_mem[(local_y - 1) * shared_stride + local_x - 1] * -1 +
            shared_mem[(local_y - 1) * shared_stride + local_x + 1] * 1 +
            shared_mem[local_y * shared_stride + local_x - 1] * -2 +
            shared_mem[local_y * shared_stride + local_x + 1] * 2 +
            shared_mem[(local_y + 1) * shared_stride + local_x - 1] * -1 +
            shared_mem[(local_y + 1) * shared_stride + local_x + 1] * 1;

        vertical_result =
            shared_mem[(local_y - 1) * shared_stride + local_x - 1] * -1 +
            shared_mem[(local_y - 1) * shared_stride + local_x] * -2 +
            shared_mem[(local_y - 1) * shared_stride + local_x + 1] * -1 +
            shared_mem[(local_y + 1) * shared_stride + local_x - 1] * 1 +
            shared_mem[(local_y + 1) * shared_stride + local_x] * 2 +
            shared_mem[(local_y + 1) * shared_stride + local_x + 1] * 1;

        float magnitude = sqrtf(horizontal_result * horizontal_result + vertical_result * vertical_result);
        d_output[y * width + x] = clamp((int)magnitude);
    }
}
"""

# Compila il codice CUDA
mod = SourceModule(cuda_code)
sobel_filter_kernel = mod.get_function("sobelFilterKernel")

def process_single_image(image_path, output_folder, block_size):
    # Carica l'immagine
    image = Image.open(image_path).convert('L')
    image_np = np.array(image, dtype=np.uint8)
    height, width = image_np.shape

    # Allocazione memoria input e output
    d_input = cuda.mem_alloc(image_np.nbytes)
    d_output = cuda.mem_alloc(image_np.nbytes)
    cuda.memcpy_htod(d_input, image_np)

    # Configura blocchi e griglia
    block = block_size
    grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1], 1)
    shared_mem_size = (block[0] + 2) * (block[1] + 2)

    print(f"Esecuzione del kernel ottimizzato con block size {block}...")

    # Lancia il kernel
    start_time = time.time()
    sobel_filter_kernel(
        d_input,
        d_output,
        np.int32(width),
        np.int32(height),
        block=block,
        grid=grid,
        shared=shared_mem_size
    )
    cuda.Context.synchronize()
    elapsed_time = time.time() - start_time
    print(f"Kernel completato in {elapsed_time:.4f} secondi con block size {block}.")

    # Copia il risultato in host
    output_np = np.empty_like(image_np)
    cuda.memcpy_dtoh(output_np, d_output)

    # Salva l'immagine output
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_image_name = f"{image_name}_optimized_sobel_block_{block[0]}x{block[1]}.png"
    output_image_path = os.path.join(output_folder, output_image_name)
    output_image = Image.fromarray(output_np)
    output_image.save(output_image_path)
    print(f"Output salvato in {output_image_path}")

if __name__ == "__main__":
    input_folder = "Dataset"  # Percorso alla cartella di input
    output_folder = "Output"  # Percorso alla cartella di output

    os.makedirs(output_folder, exist_ok=True)

    # Lista delle configurazioni di blocchi da testare
    block_sizes = [(32, 16, 1), (32, 32, 1) , (64,16,1) , (128,8,1) ,(128,4,1), (256,4,1) , (256,2,1) , (512,2,1) ,(512,1,1) , (1024,1,1) ]

    # Processa ogni immagine nella cartella con diverse configurazioni di blocchi
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, image_file)
            for block_size in block_sizes:
                print(f"Processing image: {image_path} with block size {block_size}")
                process_single_image(image_path, output_folder, block_size)


