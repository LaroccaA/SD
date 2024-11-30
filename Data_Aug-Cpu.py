import os
import time
import cv2
import shutil

def rotate_image(image, angle):
    """Effettua la rotazione di un'immagine di un angolo specifico."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotate_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

def data_augmentation_cpu(input_dir, output_dir):
    """Esegue data augmentation con rotazioni su CPU."""
    # Trova tutte le immagini nella directory di input
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = len(image_files)

    if num_images == 0:
        print("Nessuna immagine trovata nella directory di input.")
        return

    print(f"Numero di immagini trovate: {num_images}")
    
    start_time = time.perf_counter()
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
            rotated_image = rotate_image(image, angle)
            output_filename = f"{os.path.splitext(filename)[0]}_rotated_{angle}.png"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, rotated_image)
            total_generated_images += 1

    end_time = time.perf_counter()

    # Stampa statistiche
    print("\n--- Report Finale ---")
    print(f"Tempo totale di elaborazione: {end_time - start_time:.6f} secondi")
    print(f"Numero immagini iniziali: {num_images}")
    print(f"Numero immagini generate: {total_generated_images}")

    # Svuota la directory di output
    print("\nEliminazione delle immagini generate...")
    shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print("Directory di output svuotata.")

if __name__ == "__main__":
    input_folder = r"Dataset"
    output_folder = r"Dataset_augmentated"
    
    # Creazione della directory di output se non esiste
    os.makedirs(output_folder, exist_ok=True)

    # Esegui il processo di data augmentation
    data_augmentation_cpu(input_folder, output_folder)
