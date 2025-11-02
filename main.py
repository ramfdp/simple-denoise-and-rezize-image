import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
plt.ion()

def ImageRezize(images_folder, size=(800, 600)):
    images = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith(('.jpg', '.png'))]
    resized_paths = []
    for image_path in images:
        image = Image.open(image_path)
        resized_image = image.resize(size)
        base, ext = os.path.splitext(image_path)
        resized_path = f"{base}_resized{ext}"
        resized_image.save(resized_path)
        print(f"Resized image saved to {resized_path} with size {size}")
        resized_paths.append(resized_path)
    return resized_paths

def denoise_and_save(image_path, output_folder):
    noisy_image = cv2.imread(image_path)
    if noisy_image is None:
        print(f"Could not read image {image_path}")
        return
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)

    median_denoised_image = cv2.medianBlur(noisy_image, 9)
    nlm_denoised_image = cv2.fastNlMeansDenoisingColored(noisy_image, None, 15, 15, 9, 30)

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    median_output = os.path.join(output_folder, f"{name}_median_denoised{ext}")
    nlm_output = os.path.join(output_folder, f"{name}_nlm_denoised{ext}")
    
    cv2.imwrite(median_output, cv2.cvtColor(median_denoised_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(nlm_output, cv2.cvtColor(nlm_denoised_image, cv2.COLOR_RGB2BGR))
    
    print(f"Denoised images saved: {median_output}, {nlm_output}")
    
    # Display
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.title('Noisy Image')
    plt.imshow(noisy_image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Median Filter Denoising')
    plt.imshow(median_denoised_image)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Non-Local Means Denoising')
    plt.imshow(nlm_denoised_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

resized_images = ImageRezize('images')

for img in resized_images:
    denoise_and_save(img, 'images_output')

input("Tekan Enter untuk keluar")