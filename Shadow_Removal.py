import numpy as np
import cv2
import os
from PIL import Image

def show_image(image, window_name="Image"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_image(image, left, right, top, bottom):
    height, width = image.shape[:2]
    if left + right >= width or top + bottom >= height:
        raise ValueError("Cropping dimensions exceed image size.")

    cropped_image = image[top:height - bottom, left:width - right]
    return cropped_image

def adjust_image(image, contrast=0, brightness=0, shadow=0, clarity=0, highlights=0, whites=0, blacks=0, sharpness=0):
    contrast = 1.0 + (contrast / 100.0)
    brightness = int((brightness / 100.0) * 255)
    shadow = int((shadow / 100.0) * 255)
    highlights = int((highlights / 100.0) * 255)
    whites = int((whites / 100.0) * 255)
    blacks = int((blacks / 100.0) * 255)
    clarity = 1.0 + (clarity / 100.0)
    sharpness = 1.0 + (sharpness / 100.0)

    adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    if shadow > 0:
        shadow_matrix = np.full_like(adjusted, shadow, dtype=np.uint8)
        adjusted = cv2.subtract(adjusted, shadow_matrix)

    if highlights > 0:
        highlight_mask = adjusted > (255 - highlights)
        adjusted[highlight_mask] = 255 - (255 - adjusted[highlight_mask]) * (1 - highlights / 255)

    if whites > 0:
        white_mask = adjusted > (255 - whites)
        adjusted[white_mask] = 255

    if blacks > 0:
        black_mask = adjusted < blacks
        adjusted[black_mask] = adjusted[black_mask] * (1 - blacks / 255)

    if clarity != 1.0:
        kernel = clarity * (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) / 2.0)
        adjusted = cv2.filter2D(adjusted, -1, kernel)

    if sharpness != 1.0:
        kernel = sharpness * (np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) / 2.0)
        adjusted = cv2.filter2D(adjusted, -1, kernel)

    return adjusted



# Process the adjusted image
def process_image_for_circles(image, threshold=200):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # No need to convert to PIL format if not required
    return binary_image  # Return the numpy array directly



input_folder = "4mm/16cm_8ml"
output_folder = "4mm/16cm_8ml"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        file_path = os.path.join(input_folder, filename)

        # Read and preprocess the image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = crop_image(image, left=700, right=500, top=0, bottom=0)
        adjusted_image = adjust_image(
            cropped_image,
            contrast=100,
            brightness=-50,
            shadow=-100,
            clarity=100,
            highlights=-100
        )
        adjusted_image = adjust_image(
            adjusted_image,
            contrast=120,
            brightness=-50
        )

        # Process the adjusted image
        final_processed_image = process_image_for_circles(adjusted_image, threshold=200)

        # Save the processed image
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, final_processed_image)
