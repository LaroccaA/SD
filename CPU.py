import os
import time
import math
import csv
import cv2
import numpy as np

def load_image_opencv(image_path):
    """Carica un'immagine con OpenCV e la converte in formato RGB."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape
    return img, width, height

def rotate_image_manual(pixels, width, height, angle):
    """Ruota un'immagine di un dato angolo senza librerie avanzate."""
    angle_rad = math.radians(angle)

    new_width = int(abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad)))
    new_height = int(abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad)))

    rotated_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    ox, oy = width // 2, height // 2
    cx, cy = new_width // 2, new_height // 2

    for x in range(new_width):
        for y in range(new_height):
            tx = int((x - cx) * math.cos(-angle_rad) - (y - cy) * math.sin(-angle_rad) + ox)
            ty = int((x - cx) * math.sin(-angle_rad) + (y - cy) * math.cos(-angle_rad) + oy)

            if 0 <= tx < width and 0 <= ty < height:
                rotated_img[y, x] = pixels[ty, tx]

    return rotated_img, new_width, new_height

def apply_blur(pixels, kernel_size):
    """Applica una convoluzione 2D con un kernel di sfocatura personalizzabile."""
    # Crea il kernel uniforme
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)

    # Ottieni dimensioni dell'immagine
    height, width, _ = pixels.shape

    # Padding per gestire i bordi
    pad = kernel_size // 2
    padded_image = np.pad(pixels, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=0)

    # Prepara l'immagine di output
    blurred_img = np.zeros_like(pixels)

    # Esegui la convoluzione
    for x in range(height):
        if x % 100 == 0:  # Mostra lo stato di avanzamento
            print(f"    Sfocatura progress: {x}/{height} righe completate")
        for y in range(width):
            # Estrai la regione da convolvere
            region = padded_image[x:x+kernel_size, y:y+kernel_size]

            # Calcola il nuovo valore per ogni canale
            for c in range(3):  # Per ogni canale (RGB)
                blurred_img[x, y, c] = np.sum(region[:, :, c] * kernel)

    return blurred_img

def process_images(input_folder, angles, report_file, output_folder):
    """Processa tutte le immagini nella cartella, ruotandole e applicando sfocatura."""
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(report_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Angle", "Kernel Size", "Time (s)"])

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)

            print(f"Caricamento immagine: {image_file}")
            img, width, height = load_image_opencv(image_path)

            print(f"Inizio elaborazione: {image_file} ({width}x{height})")

            for angle in angles:
                # Calcolo del kernel size in base alla dimensione dell'immagine
                for blur_percentage in [0.05]:  # Percentuali della dimensione più piccola
                    kernel_size = int(min(width, height) * blur_percentage)
                    if kernel_size % 2 == 0:
                        kernel_size += 1  # Assicura che sia un numero dispari

                    print(f"  Ruotando a {angle} gradi e applicando sfocatura (kernel_size={kernel_size})...")

                    start_time = time.time()

                    # Ruota l'immagine
                    rotated_img, new_width, new_height = rotate_image_manual(img, width, height, angle)

                    # Applica la sfocatura
                    blurred_rotated_img = apply_blur(rotated_img, kernel_size)

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print(f"    Operazioni completate in {elapsed_time:.2f} secondi")

                    # Salva l'immagine ruotata e sfocata
                    output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_rotated_{angle}_blurred_kernel_{kernel_size}.png")
                    blurred_rotated_img_bgr = cv2.cvtColor(blurred_rotated_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_image_path, blurred_rotated_img_bgr)

                    # Scrive le tempistiche nel CSV
                    csvwriter.writerow([image_file, angle, kernel_size, elapsed_time])

            print(f"Elaborazione completata per {image_file}\n")


if __name__ == "__main__":
    input_folder = "./Dataset"
    angles = [90]
    report_file = "./report.csv"
    output_folder = "./Processed_Images"

    process_images(input_folder, angles, report_file, output_folder)