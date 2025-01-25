import os
import pandas as pd
from Cluster_size import process_image

def process_images_in_folder(folder_path, output_csv):
    # List to store data for CSV
    data = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Add other image formats if needed
            file_path = os.path.join(folder_path, filename)
            contour_data = process_image(file_path)

            # Count the number of clusters (contours)
            num_clusters = len(contour_data)

            # Prepare the row for the current image
            row = {'image_name': filename, 'No of Clusters': num_clusters}

            # Add cluster sizes and coordinates to the row
            for i, contour in enumerate(contour_data):
                cX, cY = contour['coordinates']
                area_mm2 = contour['area_mm2']
                row[f'Cluster {i + 1} size (mm^2)'] = area_mm2
                row[f'Cluster {i + 1} coordinate'] = f'({cX}, {cY})'

            # Append the row to the data list
            data.append(row)

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)

# Example usage
if __name__ == "__main__":
    folder_path = '4mm/16cm_8ml'  # Replace with your folder path
    output_csv = 'output_data_8ml.csv'  # Desired output CSV file name
    process_images_in_folder(folder_path, output_csv)
    print(f'Data saved to {output_csv}')




