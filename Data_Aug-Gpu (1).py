import os
import time
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA Kernel per rotazione di immagini
kernel_code = """
__global__ void rotate_image(float *input, float *output, int width, int height, float *rotate_matrix) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float new_x = rotate_matrix[0] * x + rotate_matrix[1] * y + rotate_matrix[2];
        float new_y = rotate_matrix[3] * x + rotate_matrix[4] * y + rotate_matrix[5];

        int src_x = roundf(new_x);
        int src_y = roundf(new_y);

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            output[y * width + x] = input[src_y * width + src_x];
        } else {
            output[y * width + x] = 0.0f;
        }
    }
}
"""

mod = SourceModule(kernel_code)
rotate_image = mod.get_function("rotate_image")


def data_augmentation_gpu(input_dir, output_dir):
    """Esegue data augmentation con rotazioni su GPU."""
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    if num_images == 0:
        print("Nessuna immagine trovata nella directory di input.")
        return

    print(f"Numero di immagini trovate: {num_images}")

    total_start_time = time.time()  # Timer per tutto il programma

    total_generated_images = 0
    angles = list(range(360))  # Rotazioni da 0° a 359°

    # Carica immagini in memoria
    images = []
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append((filename, image))

    gpu_start_time = time.time()  # Timer per operazioni GPU

    for filename, image in images:
        image = image.astype(np.float32)  # Assicuriamoci che sia float32
        height, width = image.shape

        if image.nbytes != width * height * np.float32().nbytes:
            print(f"Errore: dimensione immagine ({image.nbytes} bytes) non corrisponde alle aspettative ({width * height * np.float32().nbytes} bytes)")
            continue

        input_gpu = cuda.mem_alloc(image.nbytes)
        output_gpu = cuda.mem_alloc(image.nbytes)
        rotate_matrix_gpu = cuda.mem_alloc(6 * np.float32().nbytes)

        try:
            # Copia immagine sulla GPU
            cuda.memcpy_htod(input_gpu, image)
        except cuda.LogicError as e:
            print(f"Errore nella copia della memoria GPU per l'immagine {filename}: {e}")
            continue

        for angle in angles:
            # Calcola matrice di rotazione
            rotate_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1).astype(np.float32).flatten()
            cuda.memcpy_htod(rotate_matrix_gpu, rotate_matrix)

            # Configura e lancia il kernel
            block = (16, 16, 1)
            grid = (int(np.ceil(width / block[0])), int(np.ceil(height / block[1])), 1)
            rotate_image(input_gpu, output_gpu, np.int32(width), np.int32(height), rotate_matrix_gpu, block=block, grid=grid)

            # Copia il risultato dalla GPU alla CPU
            output_image = np.empty_like(image, dtype=np.float32)
            cuda.memcpy_dtoh(output_image, output_gpu)

            # Salva immagine
            output_filename = f"{os.path.splitext(filename)[0]}_rotated_{angle}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, output_image.astype(np.uint8))
            total_generated_images += 1

        # Libera memoria GPU
        input_gpu.free()
        output_gpu.free()
        rotate_matrix_gpu.free()

    gpu_end_time = time.time()  # Fine delle operazioni GPU

    total_end_time = time.time()  # Fine del programma

    # Stampa statistiche
    print("\n--- Report Finale ---")
    print(f"Tempo totale di esecuzione: {total_end_time - total_start_time:.4f} secondi")
    print(f"Tempo per operazioni GPU (CPU->GPU, rotazione, GPU->CPU): {gpu_end_time - gpu_start_time:.4f} secondi")
    print(f"Numero immagini iniziali: {num_images}")
    print(f"Numero immagini generate: {total_generated_images}")

    # Svuota directory di output
    print("\nEliminazione delle immagini generate...")
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    print("Directory di output svuotata.")



if __name__ == "__main__":
    input_folder = r"Dataset"
    output_folder = r"Dataset_augmentated"

    os.makedirs(output_folder, exist_ok=True)

    data_augmentation_gpu(input_folder, output_folder)
