import os
import numpy as np
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time

# CUDA kernel con utilizzo di memoria costante
cuda_code = """
__constant__ float d_horizontal[9];
__constant__ float d_vertical[9];

__device__ inline unsigned char clamp(int value) {
    return (value < 0) ? 0 : ((value > 255) ? 255 : value);
}

__global__ void sobelFilterKernel(unsigned char* d_input, unsigned char* d_output,
                                  int width, int height, int filterSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int radius = filterSize / 2;

    if (x < width && y < height) {
        float horizontalResult = 0.0f;
        float verticalResult = 0.0f;

        for (int fy = 0; fy < filterSize; fy++) {
            for (int fx = 0; fx < filterSize; fx++) {
                int nx = x + fx - radius;
                int ny = y + fy - radius;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    unsigned char pixelValue = d_input[ny * width + nx];
                    horizontalResult += pixelValue * d_horizontal[fy * filterSize + fx];
                    verticalResult += pixelValue * d_vertical[fy * filterSize + fx];
                }
            }
        }

        float magnitude = sqrtf(horizontalResult * horizontalResult + verticalResult * verticalResult);
        d_output[y * width + x] = clamp((int)magnitude);
    }
}
"""

# Compilazione del codice CUDA
mod = SourceModule(cuda_code)
sobel_filter_kernel = mod.get_function("sobelFilterKernel")
const_horizontal = mod.get_global("d_horizontal")[0]
const_vertical = mod.get_global("d_vertical")[0]

def apply_sobel_filter(input_folder, output_folder):
    # Carica tutti i file immagine nella cartella di input
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not input_files:
        raise FileNotFoundError(f"No images found in the folder: {input_folder}")

    # Definizione delle maschere Sobel
    sobel_horizontal = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1], dtype=np.float32)
    sobel_vertical = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1], dtype=np.float32)
    filter_size = 3

    # Copia delle maschere nella memoria costante
    cuda.memcpy_htod(const_horizontal, sobel_horizontal)
    cuda.memcpy_htod(const_vertical, sobel_vertical)

    # Processa ciascuna immagine
    for image_file in input_files:
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing image: {image_path}")

        # Carica l'immagine in scala di grigi
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, dtype=np.uint8)
        height, width = image_np.shape

        # Allocazione memoria per input e output
        d_input = cuda.mem_alloc(image_np.nbytes)
        d_output = cuda.mem_alloc(image_np.nbytes)
        cuda.memcpy_htod(d_input, image_np)

        # Test di configurazioni specifiche di blocchi
        for block_size_x, block_size_y in [(16, 16)]:
            block = (block_size_x, block_size_y, 1)
            grid = ((width + block_size_x - 1) // block_size_x, (height + block_size_y - 1) // block_size_y, 1)

            print(f"  Running kernel with block size ({block_size_x}, {block_size_y})...")

            # Lancia il kernel
            start_time = time.time()
            sobel_filter_kernel(
                d_input,
                d_output,
                np.int32(width),
                np.int32(height),
                np.int32(filter_size),
                block=block,
                grid=grid
            )
            cuda.Context.synchronize()
            elapsed_time = time.time() - start_time
            print(f"    Completed in {elapsed_time:.4f} seconds.")

            # Copia il risultato dalla GPU alla CPU
            output_np = np.empty_like(image_np)
            cuda.memcpy_dtoh(output_np, d_output)

            # Salva l'immagine di output per questa configurazione
            output_image_name = f"{os.path.splitext(image_file)[0]}_block_{block_size_x}x{block_size_y}.png"
            output_image_path = os.path.join(output_folder, output_image_name)
            output_image = Image.fromarray(output_np)
            output_image.save(output_image_path)
            print(f"    Output saved as {output_image_path}")

if __name__ == "__main__":
    input_folder = "Dataset"  # Percorso alla cartella di input
    output_folder = "Output"  # Percorso alla cartella di output

    os.makedirs(output_folder, exist_ok=True)

    # Applica il filtro Sobel con configurazione 2D2D
    print("Running with 2D2D configuration...")
    apply_sobel_filter(input_folder, output_folder)
