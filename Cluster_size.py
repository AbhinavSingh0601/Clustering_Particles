import cv2
import numpy as np

def process_image(file_name):
    # Load the image
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to create a binary image
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find the contours of the dots
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Conversion factor from pixels to mm²
    area_per_pixel = 16 / (96 * 96)  # mm² per pixel

    # List to store the results
    results = []

    # Iterate through the contours
    for contour in contours:
        # Calculate the area of the contour in pixels
        area_pixels = cv2.contourArea(contour)

        # Convert the area to mm²
        area_mm2 = area_pixels * area_per_pixel

        # Only consider contours with area greater than a threshold
        if area_mm2 > 1:  # You can adjust this threshold as needed
            # Get the centroid of the contour for labeling
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # Append the results as a dictionary
            if cX >= 150:
                results.append({
                    'coordinates': (cX, cY),
                    'area_mm2': area_mm2
                })

    return results
import cv2
import os

def display_image_with_annotations(file_name, contour_data, output_directory):
    # Load the original image in color for display
    image = cv2.imread(file_name)

    # Load the grayscale image to find contours
    gray_image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green

    # Iterate through the contour data to annotate the image
    for contour in contour_data:
        cX, cY = contour['coordinates']
        area_mm2 = contour['area_mm2']

        # Highlight the center of the contour
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)  # Draw a red circle at the centroid

        # Label the area on the output image
        label = f'{area_mm2:.2f} mm²'
        cv2.putText(image, label, (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Display the (x, y) coordinates
        coord_label = f'({cX}, {cY})'
        cv2.putText(image, coord_label, (cX + 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Show the annotated image
    cv2.imshow('Annotated Image', image)

    # Save the annotated image to the specified output directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the directory if it doesn't exist

    # Extract the filename from the input file path
    base_filename = os.path.basename(file_name)
    output_file_path = os.path.join(output_directory, f'annotated_{base_filename}')

    # Save the image
    cv2.imwrite(output_file_path, image)
    print(f'Annotated image saved to: {output_file_path}')

    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
# Example usage
    file_name = '4mm/16cm_0ml/fi_000000_processed/0004054.jpg'  # Replace with your image path
    contour_data = process_image(file_name)

    # Print the results
    for contour in contour_data:
        print(f'Coordinates: {contour["coordinates"]}, Area: {contour["area_mm2"]:.4f} mm²')


    display_image_with_annotations(file_name, contour_data, 'output_images')