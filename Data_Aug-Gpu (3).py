import os
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Kernel CUDA ottimizzato con memoria condivisa e parallelizzazione degli angoli
kernel_code = """
__global__ void rotate_image(float *input, float *output, int width, int height, float *angles) {
    extern __shared__ float tile[];
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int angle_idx = blockIdx.z; // Indice della terza dimensione (angolo)

    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    // Calcolo centro immagine
    float centerX = width / 2.0;
    float centerY = height / 2.0;

    // Caricamento nella memoria condivisa
    if (x < width && y < height) {
        tile[local_y * blockDim.x + local_x] = input[y * width + x];
    }
    __syncthreads();

    if (x < width && y < height) {
        // Calcolo nuova posizione
        float angle = angles[angle_idx];
        float newX = cos(angle) * (x - centerX) - sin(angle) * (y - centerY) + centerX;
        float newY = sin(angle) * (x - centerX) + cos(angle) * (y - centerY) + centerY;

        int srcX = (int)newX;
        int srcY = (int)newY;

        if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
            output[(angle_idx * height + y) * width + x] = tile[(srcY % blockDim.y) * blockDim.x + (srcX % blockDim.x)];
        } else {
            output[(angle_idx * height + y) * width + x] = 0.0f; // Riempimento con nero
        }
    }
}
"""

mod = SourceModule(kernel_code)
rotate_image_kernel = mod.get_function("rotate_image")

def data_augmentation_gpu(input_dir, output_dir):
    """Esegue data augmentation con rotazioni su GPU."""
    # Trova tutte le immagini nella directory di input
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    if num_images == 0:
        print("Nessuna immagine trovata nella directory di input.")
        return

    print(f"Numero di immagini trovate: {num_images}")

    # Carica la prima immagine per determinare la dimensione
    sample_image = cv2.imread(os.path.join(input_dir, image_files[0]), cv2.IMREAD_GRAYSCALE)
    if sample_image is None:
        print("Errore nel leggere l'immagine campione.")
        return

    height, width = sample_image.shape
    print(f"Dimensioni immagine: {height}x{width}")
    image_memory_size = height * width * np.float32().nbytes
    output_memory_size = height * width * len(range(360)) * np.float32().nbytes

    # Inizializza la memoria sulla GPU
    input_gpu = cuda.mem_alloc(image_memory_size)
    output_gpu = cuda.mem_alloc(output_memory_size)
    angles_gpu = cuda.mem_alloc(360 * np.float32().nbytes)

    # Prepara gli angoli in radianti
    angles = np.radians(np.arange(360)).astype(np.float32)
    cuda.memcpy_htod(angles_gpu, angles)

    # Configurazioni di test per blocchi
    block_sizes = [(8, 8, 1), (16, 16, 1), (32, 32, 1)]

    for block in block_sizes:
        # Calcola la griglia per l'attuale configurazione di blocchi
        grid = (
            int(np.ceil(width / block[0])),
            int(np.ceil(height / block[1])),
            len(angles)  # Terza dimensione per gli angoli
        )
        shared_memory_size = block[0] * block[1] * np.float32().nbytes
        print(f"\nTesting configuration - Block: {block}, Grid: {grid}, Shared Memory: {shared_memory_size} bytes")

        total_time = 0.0

        # Benchmark per ogni immagine
        for filename in image_files:
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Errore nel leggere l'immagine {filename}, la ignoro.")
                continue

            # Assicurati che l'immagine sia float32
            image = image.astype(np.float32)
            print(f"Elaborando immagine: {filename}, Dimensioni: {image.shape}, Memoria: {image.nbytes} bytes")

            # Copia immagine nella GPU
            cuda.memcpy_htod(input_gpu, image)

            # Lancia il kernel
            start = time.perf_counter()
            rotate_image_kernel(
                input_gpu, 
                output_gpu, 
                np.int32(width), 
                np.int32(height), 
                angles_gpu, 
                block=block, 
                grid=grid, 
                shared=shared_memory_size
            )
            cuda.Context.synchronize()
            end = time.perf_counter()

            # Copia il risultato dalla GPU
            output_images = np.empty((360, height, width), dtype=np.float32)
            cuda.memcpy_dtoh(output_images, output_gpu)

            # Salva tutte le immagini ruotate
            for angle_idx, rotated_image in enumerate(output_images):
                output_filename = f"{os.path.splitext(filename)[0]}_rotated_{angle_idx}_block{block[0]}.png"
                cv2.imwrite(os.path.join(output_dir, output_filename), rotated_image)

            total_time += (end - start)

        print(f"Tempo totale per configurazione {block}: {total_time:.6f} secondi")
        print(f"Tempo medio per immagine: {total_time / num_images:.6f} secondi\n")

    # Svuota la directory di output
    print("\nEliminazione delle immagini generate...")
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, f))
    print("Directory di output svuotata.")

if __name__ == "__main__":
    input_folder = r"C:/Users/andre/Desktop/Uni/Appunti/Sistemi Digitali/Progetto/Dataset"
    output_folder = r"C:/Users/andre/Desktop/Uni/Appunti/Sistemi Digitali/Progetto/Dataset_augmentated"
    
    # Creazione della directory di output se non esiste
    os.makedirs(output_folder, exist_ok=True)
    
    # Timer globale
    overall_start = time.time()
    
    # Esegui il processo di data augmentation
    data_augmentation_gpu(input_folder, output_folder)
    
    overall_end = time.time()
    print(f"\nTempo totale del programma (completo): {overall_end - overall_start:.4f} secondi")
