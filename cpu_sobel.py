import numpy as np
from PIL import Image
import os
import time

def clamp(value):
    return max(0, min(255, value))

def apply_convolution(image, kernel):
    height, width = image.shape
    kernel_size = kernel.shape[0]
    radius = kernel_size // 2

    output = np.zeros_like(image, dtype=np.float32)

    print(f"Applying convolution with kernel size {kernel_size}...")

    for y in range(height):
        for x in range(width):
            value = 0.0
            for ky in range(kernel_size):
                for kx in range(kernel_size):
                    nx = x + kx - radius
                    ny = y + ky - radius

                    if 0 <= nx < width and 0 <= ny < height:
                        value += image[ny, nx] * kernel[ky, kx]
            output[y, x] = clamp(int(value))

        if y % (height // 10) == 0:  # Print progress every 10% of rows
            print(f"  Progress: {y}/{height} rows processed...")

    print("Convolution completed.")

    return output.astype(np.uint8)

def combine_filters(horizontal, vertical):
    print("Combining horizontal and vertical filter results...")
    combined = np.sqrt(horizontal.astype(np.float32) ** 2 + vertical.astype(np.float32) ** 2)
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    print("Combination completed.")
    return combined

def main():
    input_folder = "Dataset"
    output_folder = "Output"
    os.makedirs(output_folder, exist_ok=True)

    sobel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    file_list = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_files = len(file_list)

    print(f"Starting processing {total_files} files...")

    for i, filename in enumerate(file_list, start=1):
        print(f"Processing file {i}/{total_files}: {filename}...")
        input_path = os.path.join(input_folder, filename)
        image = Image.open(input_path).convert('L')
        image_np = np.array(image)

        start_time = time.time()
        print("  Applying horizontal Sobel filter...")
        output_horizontal = apply_convolution(image_np, sobel_horizontal)
        print("  Horizontal filter applied.")

        print("  Applying vertical Sobel filter...")
        output_vertical = apply_convolution(image_np, sobel_vertical)
        print("  Vertical filter applied.")

        print("  Combining filters...")
        combined_output = combine_filters(output_horizontal, output_vertical)
        print("  Filters combined.")

        elapsed_time = time.time() - start_time
        print(f"  Processing completed in {elapsed_time:.4f} seconds.")

        combined_output_image = Image.fromarray(combined_output)
        combined_output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_sobel_combined.png")
        combined_output_image.save(combined_output_path)

        print(f"  Combined result saved to {combined_output_path}.")

    print("All files processed successfully.")

if __name__ == "__main__":
    main()



