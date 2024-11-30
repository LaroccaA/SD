import os
import time
import cv2
import shutil
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Kernel CUDA per la rotazione delle immagini
kernel_code = """
__global__ void rotate_image(float *input, float *output, int width, int height, float *rotate_matrix) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        // Calcola la nuova posizione con la matrice di rotazione
        float new_x = rotate_matrix[0] * x + rotate_matrix[1] * y + rotate_matrix[2];
        float new_y = rotate_matrix[3] * x + rotate_matrix[4] * y + rotate_matrix[5];

        int src_x = roundf(new_x);
        int src_y = roundf(new_y);

        if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
            output[y * width + x] = input[src_y * width + src_x];
        } else {
            output[y * width + x] = 0.0f; // Riempimento con nero
        }
    }
}
"""

mod = SourceModule(kernel_code)
rotate_image = mod.get_function("rotate_image")

def augment_image_gpu(image, angle):
    """Effettua la rotazione di un'immagine su GPU per un dato angolo."""
    height, width = image.shape

    # Assicurati che l'immagine sia float32
    image = image.astype(np.float32)

    # Calcola la matrice di rotazione
    center = (width / 2, height / 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32).flatten()

    # Alloca memoria sulla GPU
    input_gpu = cuda.mem_alloc(image.nbytes)
    output_gpu = cuda.mem_alloc(image.nbytes)
    rotate_matrix_gpu = cuda.mem_alloc(rotate_matrix.nbytes)

    # Copia i dati sulla GPU
    cuda.memcpy_htod(input_gpu, image)
    cuda.memcpy_htod(rotate_matrix_gpu, rotate_matrix)

    # Configura il kernel
    block = (16, 16, 1)
    grid = (int(np.ceil(width / block[0])), int(np.ceil(height / block[1])), 1)

    # Lancia il kernel
    rotate_image(input_gpu, output_gpu, np.int32(width), np.int32(height), rotate_matrix_gpu, block=block, grid=grid)

    # Copia il risultato dalla GPU
    output_image = np.empty_like(image)
    cuda.memcpy_dtoh(output_image, output_gpu)

    return output_image

def data_augmentation_gpu(input_dir, output_dir):
    """Esegue data augmentation con rotazioni su GPU."""
    # Trova tutte le immagini nella directory di input
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    if num_images == 0:
        print("Nessuna immagine trovata nella directory di input.")
        return

    print(f"Numero di immagini trovate: {num_images}")

    start_time = time.time()
    total_generated_images = 0

    # Elaborazione di ogni immagine
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Errore nel leggere l'immagine {filename}, la ignoro.")
            continue

        # Genera 360 rotazioni
        for angle in range(360):
            rotated_image = augment_image_gpu(image, angle)
            output_filename = f"{os.path.splitext(filename)[0]}_rotated_{angle}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, rotated_image)
            total_generated_images += 1

    end_time = time.time()

    # Stampa statistiche
    print("\n--- Report Finale ---")
    print(f"Tempo totale di elaborazione: {end_time - start_time:.2f} secondi")
    print(f"Numero immagini iniziali: {num_images}")
    print(f"Numero immagini generate: {total_generated_images}")

    # Svuota la directory di output
    print("\nEliminazione delle immagini generate...")
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Directory di output svuotata.")

if __name__ == "__main__":
    input_folder = r"C:\Users\andre\Desktop\Uni\Appunti\Sistemi Digitali\Progetto\Dataset"
    output_folder = r"C:\Users\andre\Desktop\Uni\Appunti\Sistemi Digitali\Progetto\Dataset_augmentated"

    # Creazione della directory di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Esegui il processo di data augmentation
    data_augmentation_gpu(input_folder, output_folder)
