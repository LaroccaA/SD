import os
import time
import math
import csv
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Kernel CUDA scritto in CUDA C per la rotazione
CUDA_ROTATE_KERNEL = """
__global__ void rotate_image_kernel(unsigned char* pixels, unsigned char* rotated_img, int width, int height, float angle_rad, int ox, int oy, int cx, int cy, int new_width, int new_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < new_width && y < new_height) {
        int tx = (int)((x - cx) * cosf(-angle_rad) - (y - cy) * sinf(-angle_rad) + ox);
        int ty = (int)((x - cx) * sinf(-angle_rad) + (y - cy) * cosf(-angle_rad) + oy);

        if (tx >= 0 && tx < width && ty >= 0 && ty < height) {
            int rotated_idx = (y * new_width + x) * 3;
            int original_idx = (ty * width + tx) * 3;
            rotated_img[rotated_idx] = pixels[original_idx];
            rotated_img[rotated_idx + 1] = pixels[original_idx + 1];
            rotated_img[rotated_idx + 2] = pixels[original_idx + 2];
        }
    }
}
"""

# Kernel CUDA per applicare il blur (convoluzione 2D) con memoria condivisa
CUDA_BLUR_KERNEL_SHARED = """
__global__ void blur_kernel_shared(unsigned char* pixels, unsigned char* blurred_img, int width, int height, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int half_kernel = kernel_size / 2;

    // Dimensione della memoria condivisa con padding per il kernel
    extern __shared__ unsigned char shared_mem[];

    unsigned char* shared_r = shared_mem;
    unsigned char* shared_g = &shared_mem[(blockDim.x + 2 * half_kernel) * (blockDim.y + 2 * half_kernel)];
    unsigned char* shared_b = &shared_g[(blockDim.x + 2 * half_kernel) * (blockDim.y + 2 * half_kernel)];

    int shared_width = blockDim.x + 2 * half_kernel;

    // Calcolo della posizione nella memoria globale
    int global_idx = (y * width + x) * 3;

    // Calcolo della posizione nella memoria condivisa
    int shared_x = threadIdx.x + half_kernel;
    int shared_y = threadIdx.y + half_kernel;
    int shared_idx = shared_y * shared_width + shared_x;

    // Caricamento nella memoria condivisa
    if (x < width && y < height) {
        shared_r[shared_idx] = pixels[global_idx];
        shared_g[shared_idx] = pixels[global_idx + 1];
        shared_b[shared_idx] = pixels[global_idx + 2];
    } else {
        shared_r[shared_idx] = 0;
        shared_g[shared_idx] = 0;
        shared_b[shared_idx] = 0;
    }

    // Caricamento dei bordi nella memoria condivisa
    if (threadIdx.x < half_kernel) {
        int left_x = max(x - half_kernel, 0);
        int left_idx = (y * width + left_x) * 3;
        shared_r[shared_y * shared_width + threadIdx.x] = pixels[left_idx];
        shared_g[shared_y * shared_width + threadIdx.x] = pixels[left_idx + 1];
        shared_b[shared_y * shared_width + threadIdx.x] = pixels[left_idx + 2];
    }

    if (threadIdx.x >= blockDim.x - half_kernel) {
        int right_x = min(x + half_kernel, width - 1);
        int right_idx = (y * width + right_x) * 3;
        shared_r[shared_y * shared_width + threadIdx.x + 2 * half_kernel] = pixels[right_idx];
        shared_g[shared_y * shared_width + threadIdx.x + 2 * half_kernel] = pixels[right_idx + 1];
        shared_b[shared_y * shared_width + threadIdx.x + 2 * half_kernel] = pixels[right_idx + 2];
    }

    if (threadIdx.y < half_kernel) {
        int top_y = max(y - half_kernel, 0);
        int top_idx = (top_y * width + x) * 3;
        shared_r[threadIdx.y * shared_width + shared_x] = pixels[top_idx];
        shared_g[threadIdx.y * shared_width + shared_x] = pixels[top_idx + 1];
        shared_b[threadIdx.y * shared_width + shared_x] = pixels[top_idx + 2];
    }

    if (threadIdx.y >= blockDim.y - half_kernel) {
        int bottom_y = min(y + half_kernel, height - 1);
        int bottom_idx = (bottom_y * width + x) * 3;
        shared_r[(threadIdx.y + 2 * half_kernel) * shared_width + shared_x] = pixels[bottom_idx];
        shared_g[(threadIdx.y + 2 * half_kernel) * shared_width + shared_x] = pixels[bottom_idx + 1];
        shared_b[(threadIdx.y + 2 * half_kernel) * shared_width + shared_x] = pixels[bottom_idx + 2];
    }

    __syncthreads();

    // Applicazione del filtro
    if (x < width && y < height) {
        float r = 0, g = 0, b = 0;
        int count = 0;

        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int idx = (shared_y + ky) * shared_width + (shared_x + kx);
                r += shared_r[idx];
                g += shared_g[idx];
                b += shared_b[idx];
                count++;
            }
        }

        blurred_img[global_idx] = r / count;
        blurred_img[global_idx + 1] = g / count;
        blurred_img[global_idx + 2] = b / count;
    }
}
"""

# Funzione per caricare un'immagine con OpenCV in formato RGB
def load_image_opencv(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    return img, width, height

# Funzione per ruotare l'immagine usando PyCUDA
def rotate_image_cuda(pixels, width, height, angle, threads_per_block):
    angle_rad = math.radians(angle)

    # Calcola le dimensioni della nuova immagine
    new_width = int(abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad)))
    new_height = int(abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad)))

    # Creazione immagine vuota per il risultato
    rotated_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Calcola i centri
    ox, oy = width // 2, height // 2
    cx, cy = new_width // 2, new_height // 2

    # Compilazione del kernel CUDA per la rotazione
    module = SourceModule(CUDA_ROTATE_KERNEL)
    rotate_image_kernel = module.get_function("rotate_image_kernel")

    # Allocazione della memoria sulla GPU
    d_pixels = cuda.mem_alloc(pixels.nbytes)
    d_rotated_img = cuda.mem_alloc(rotated_img.nbytes)

    cuda.memcpy_htod(d_pixels, pixels)
    cuda.memcpy_htod(d_rotated_img, rotated_img)

    # Configurazione blocchi e griglie
    blocks_per_grid_x = math.ceil(new_width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(new_height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, 1)

    # Esecuzione del kernel
    rotate_image_kernel(
        d_pixels,
        d_rotated_img,
        np.int32(width),
        np.int32(height),
        np.float32(angle_rad),
        np.int32(ox),
        np.int32(oy),
        np.int32(cx),
        np.int32(cy),
        np.int32(new_width),
        np.int32(new_height),
        block=threads_per_block,
        grid=blocks_per_grid
    )

    # Copia del risultato dalla GPU
    cuda.memcpy_dtoh(rotated_img, d_rotated_img)
    return rotated_img, new_width, new_height

# Funzione per applicare il blur all'immagine usando CUDA con memoria condivisa
def blur_image_cuda(pixels, width, height, kernel_size, threads_per_block):
    blurred_img = np.zeros_like(pixels)

    # Compilazione del kernel CUDA per il blur
    module = SourceModule(CUDA_BLUR_KERNEL_SHARED)
    blur_kernel_shared = module.get_function("blur_kernel_shared")

    # Allocazione della memoria sulla GPU
    d_pixels = cuda.mem_alloc(pixels.nbytes)
    d_blurred_img = cuda.mem_alloc(blurred_img.nbytes)

    cuda.memcpy_htod(d_pixels, pixels)
    cuda.memcpy_htod(d_blurred_img, blurred_img)

    # Configurazione blocchi e griglie
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, 1)

    max_shared_mem = cuda.Device(0).get_attribute(cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK)
    shared_mem_size = (threads_per_block[0] + 2 * (kernel_size // 2)) * (threads_per_block[1] + 2 * (kernel_size // 2)) * 3

    # Calcolo dinamico del kernel_size massimo accettabile
    if shared_mem_size > max_shared_mem:
        max_kernel_size = int(math.sqrt(max_shared_mem / (threads_per_block[0] + 2) / (threads_per_block[1] + 2) / 3))
        kernel_size = min(kernel_size, max_kernel_size * 2 + 1)
        shared_mem_size = (threads_per_block[0] + 2 * (kernel_size // 2)) * (threads_per_block[1] + 2 * (kernel_size // 2)) * 3

    print(f"Grid: {blocks_per_grid}, Block: {threads_per_block}, Shared Memory: {shared_mem_size}, Kernel Size: {kernel_size}")

    print(f"Esecuzione kernel blur con shared_mem_size: {shared_mem_size}")
# Esecuzione del kernel
    blur_kernel_shared(
        d_pixels,
        d_blurred_img,
        np.int32(width),
        np.int32(height),
        np.int32(kernel_size),
        block=threads_per_block,
        grid=blocks_per_grid,
        shared=shared_mem_size
    )

    # Copia del risultato dalla GPU
    cuda.memcpy_dtoh(blurred_img, d_blurred_img)

    # Salva immagine sfocata per debug
    debug_blur_path = "./debug_blurred_image.png"
    cv2.imwrite(debug_blur_path, cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR))
    print(f"Immagine sfocata salvata per debug: {debug_blur_path}")
    print(f" Ho eseguito il kernel di sfocature")

    return blurred_img

# Funzione per processare le immagini e generare un report
def process_images(input_folder, angles, report_file, output_folder):
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    threads_per_block_options = [(16, 16, 1)]

    with open(report_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Resolution", "Angle", "Threads Per Block", "Time (s)"])

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)

            print(f"Caricamento immagine: {image_file}")
            img, width, height = load_image_opencv(image_path)

            print(f"Inizio elaborazione: {image_file} ({width}x{height})")

            generated_images = 0

            for threads_per_block in threads_per_block_options:
                for angle in angles:
                    print(f"  Ruotando a {angle} gradi con blocchi {threads_per_block}...")

                    start_time = time.time()
                    rotated_img, new_width, new_height = rotate_image_cuda(img, width, height, angle, threads_per_block)

                    # Applichiamo il blur
                    kernel_size = 129  # Esempio con un kernel grande
                    blurred_img = blur_image_cuda(rotated_img, new_width, new_height, kernel_size, threads_per_block)

                    end_time = time.time()

                    elapsed_time = end_time - start_time

                    csvwriter.writerow([image_file, f"{width}x{height}", angle, threads_per_block, elapsed_time])

                    rotated_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_rotated_{angle}_{threads_per_block[0]}x{threads_per_block[1]}.png")
                    blurred_img_bgr = cv2.cvtColor(blurred_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(rotated_image_path, blurred_img_bgr)

                    generated_images += 1

            print(f"Immagini generate per {image_file}: {generated_images}\n")

if __name__ == "__main__":
    input_folder = "./Dataset"
    angles = [90]
    report_file = "./report.csv"
    output_folder = "./Processed_Images"

    process_images(input_folder, angles, report_file, output_folder)
