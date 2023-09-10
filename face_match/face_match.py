"""
    Author: Ahmed Sobhi
    Department: Data Science
    Created_at: 2023-09-10
    Objective: Inference for Face Recognition.
"""
import warnings
warnings.filterwarnings('ignore')

import recognition
import argparse

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Inference for Face Recognition.')

# Define command-line arguments with default values
parser.add_argument('--mode', default='single', help='Handling either [single, multi] images in infrence (default: signle)')
parser.add_argument('--img_dir', default='data/test/', help='Input directory path (default: /data/test/)')
parser.add_argument('--img_name', default=None, help='Input Image Name, ex: image.jpg (default: None)')
parser.add_argument('--label_path', default='data/label/', help='Label directory path (default: /data/label)')
parser.add_argument('--recognition_model', default='ArcFace', help='Recognition Model (default: ArcFace)')
parser.add_argument('--detection_model', default='retinaface', help='Detectiom Model (default: retinaface)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the argument values
mode = args.mode
img_path = args.img_dir
img_name = args.img_name
label_path = args.label_path

# Defining models
face_recognition_model = args.recognition_model
face_detector_model = args.detection_model

if __name__ == '__main__':

    if mode == 'single':
        _ = recognition.single_image_recognition(
                img_path=img_path,
                img_name=img_name,
                label_path=label_path,
                model_name=face_recognition_model,
                detector_backend=face_detector_model
            )
    
    elif mode == 'multi':
        _ = recognition.multi_image_recognition(
                img_path=img_path,
                label_path=label_path,
                model_name=face_recognition_model,
                detector_backend=face_detector_model
            )
