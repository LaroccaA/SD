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

    # Calcola le dimensioni della nuova immagine
    new_width = int(abs(width * math.cos(angle_rad)) + abs(height * math.sin(angle_rad)))
    new_height = int(abs(height * math.cos(angle_rad)) + abs(width * math.sin(angle_rad)))

    # Creazione di una matrice vuota per la nuova immagine
    rotated_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    # Centro della vecchia e della nuova immagine
    ox, oy = width // 2, height // 2
    cx, cy = new_width // 2, new_height // 2

    # Rotazione pixel per pixel
    for x in range(new_width):
        if x % 100 == 0:
            print(f"  Progress: x = {x}/{new_width}")
        for y in range(new_height):
            # Coordinate inverse
            tx = int((x - cx) * math.cos(-angle_rad) - (y - cy) * math.sin(-angle_rad) + ox)
            ty = int((x - cx) * math.sin(-angle_rad) + (y - cy) * math.cos(-angle_rad) + oy)

            # Controlla se le coordinate sono all'interno della vecchia immagine
            if 0 <= tx < width and 0 <= ty < height:
                rotated_img[y, x] = pixels[ty, tx]

    return rotated_img, new_width, new_height

def process_images(input_folder, angles, report_file, output_folder):
    """Processa tutte le immagini nella cartella per le rotazioni specificate."""
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff'))]

    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Apri il file CSV per scrivere il report
    with open(report_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image Name", "Resolution", "Angle", "Time (s)"])

        for image_file in image_files:
            image_path = os.path.join(input_folder, image_file)

            # Carica l'immagine usando OpenCV
            print(f"Caricamento immagine: {image_file}")
            img, width, height = load_image_opencv(image_path)

            print(f"Inizio elaborazione: {image_file} ({width}x{height})")

            generated_images = 0

            for angle in angles:
                print(f"  Ruotando a {angle} gradi...")

                start_time = time.time()
                rotated_img, new_width, new_height = rotate_image_manual(img, width, height, angle)
                end_time = time.time()

                elapsed_time = end_time - start_time
                print(f"    Rotazione completata in {elapsed_time:.2f} secondi")

                # Salva il risultato nel file CSV
                csvwriter.writerow([image_file, f"{width}x{height}", angle, elapsed_time])

                # Salva l'immagine ruotata in formato PNG
                rotated_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_rotated_{angle}.png")
                rotated_img_bgr = cv2.cvtColor(rotated_img, cv2.COLOR_RGB2BGR)  # Converti da RGB a BGR per OpenCV
                cv2.imwrite(rotated_image_path, rotated_img_bgr)

                generated_images += 1

            print(f"Immagini generate per {image_file}: {generated_images}\n")

if __name__ == "__main__":
    # Cartella di input contenente le immagini
    input_folder = "./Dataset"

    # Angoli di rotazione richiesti
    angles = [45, 90, 135, 180, 225, 270, 315, 360]

    # Percorso del file di report CSV
    report_file = "./report.csv"

    # Cartella di output per salvare le immagini ruotate
    output_folder = "./Rotated_Images"

    # Processa le immagini
    process_images(input_folder, angles, report_file, output_folder)
