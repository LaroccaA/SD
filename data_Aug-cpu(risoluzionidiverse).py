
import os
import time
import cv2
import shutil

def rotate_image(image, angle):
    """Effettua la rotazione di un'immagine di un angolo specifico senza alterarne la risoluzione."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return rotated_image

def measure_rotation_times(input_dir):
    """Misura il tempo di rotazione per immagini di diverse risoluzioni con rotazioni multiple mantenendo la risoluzione originale."""
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print("Nessuna immagine trovata nella directory di input.")
        return

    output_dir = "Output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"Numero di immagini trovate: {len(image_files)}")

    report = []

    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Errore nel leggere l'immagine {filename}, la ignoro.")
            continue

        h, w = image.shape[:2]
        print(f"DEBUG: Risoluzione caricata di {filename}: {w}x{h}")
        image_report = [f"Immagine: {filename} | Risoluzione: {w}x{h}"]

        for angle in range(30, 360, 30):
            start_time = time.perf_counter()
            rotated_image = rotate_image(image, angle)
            end_time = time.perf_counter()

            output_path = os.path.join(output_dir, f"{filename}_rotated_{angle}.png")
            cv2.imwrite(output_path, rotated_image)

            duration = end_time - start_time
            image_report.append(f"Angolo: {angle}° | Tempo di rotazione: {duration:.6f} secondi")

        report.extend(image_report)

    # Stampa il report finale
    print("\n--- Report Finale ---")
    for entry in report:
        print(entry)

if __name__ == "__main__":
    input_folder = r"Dataset"
    measure_rotation_times(input_folder)
