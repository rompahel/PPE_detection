import os
import pandas as pd

# Define your classes in the same order as used in YOLO
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'Machinery', 'Vehicle']

def convert_yolo_to_multilabel(images_folder, labels_folder, output_csv):
    data = []
    
    # Process all label files
    for label_file in os.listdir(labels_folder):
        if label_file.endswith('.txt'):
            # Initialize binary vector for this image
            label_vector = [0] * len(classes)
            
            # Read YOLO label file
            with open(os.path.join(labels_folder, label_file), 'r') as f:
                for line in f:
                    class_id = int(line.split()[0])  # Class ID is the first value in YOLO format
                    label_vector[class_id] = 1  # Set corresponding class to 1
            
            # Match label file to image file
            image_file = label_file.replace('.txt', '.jpg')  # Assuming images are .jpg
            image_path = os.path.join(images_folder, image_file)
            
            # Add to dataset
            data.append([image_path] + label_vector)
    
    # Create DataFrame
    column_names = ['image_path'] + classes
    df = pd.DataFrame(data, columns=column_names)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Multi-label dataset saved to {output_csv}")

# Example usage
convert_yolo_to_multilabel(
    images_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\train\images',
    labels_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\train\labels',
    output_csv='train_labels.csv'
)
convert_yolo_to_multilabel(
    images_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\valid\images',
    labels_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\valid\labels',
    output_csv='valid_labels.csv'
)
convert_yolo_to_multilabel(
    images_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\test\images',
    labels_folder=r'D:\Kuliah\Bangkit\PPEDetection\css-data\test\labels',
    output_csv='test_labels.csv'
)
