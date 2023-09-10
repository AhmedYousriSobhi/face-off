# Importing Required Libararies & Packages
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np, pandas as pd
import cv2
from deepface import DeepFace

# To run on CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Functions Definitions
def custom_draw_on(img, faces, label_lst):
    """
        Function used to draw rectangular box around the detected faces with their labels.
    """
    dimg = img.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face['facial_area']
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box['x'], box['y']), (box['x']+box['w'], box['y']+box['h']), color, 2)
        cv2.putText(dimg, label_lst[i], (box['x']-1, box['y']-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)

    return dimg


def load_label_csv(label_path)->pd.DataFrame:
    """
        Function used to load csv label file
    """
    # Load Label CSV file
    # Step: Searching for a csv file locatted in this directory
    for file in os.listdir(label_path):
        if file.endswith(".csv"): 
            csv_path = os.path.join(label_path, file)

    # Step: load the csv label file
    return pd.read_csv(csv_path)


# Single Image Recognition Inference
def single_image_recognition(
        img_path, 
        img_name,
        label_path,
        model_name,
        detector_backend
        ):
    """
        Function used to detect faces and recognition them in a single Image
    """

    # Load label Database
    df_label = load_label_csv(label_path=label_path)

    detected_faces = DeepFace.extract_faces(
        img_path=img_path+img_name,
        detector_backend=detector_backend
    )

    found_faces = DeepFace.find(img_path=img_path+img_name,
                db_path=label_path,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=True)


    label_lst = []

    for found_face in found_faces:

        if len(found_face) != 0:
            # Face was detected
            img_label = 'UnKown'
            
            # checking the ArcFace_cosine threshold
            if found_face['ArcFace_cosine'].values[0]> 0.25:
                img_label_path = found_face['identity'].values[0]
                img_label = df_label[df_label.image_name == img_label_path[img_label_path.rfind('/') + 1:]]['label'].values[0]
                
            label_lst.append(img_label)

        else:
            label_lst.append('Unkown')

    img = cv2.imread(img_path+img_name)

    rimg= custom_draw_on(img, detected_faces, label_lst)


    # Output the labeled Image
    cv2.imwrite(f"../report/plots/{img_name}", rimg)

    print(f"Recognitioned Faces in {img_name} are: {label_lst}. Exported image with detected faces at: ../report/plots/{img_name}")

    # To indicate the function terminated successfully
    return True


# Directory Image Recognition Inference
def multi_image_recognition(
        img_path, 
        label_path,
        model_name,
        detector_backend
        ):
    """
        Function used to detect faces and recognition them in on multiple Image
    """

    # Load label Database
    df_label = load_label_csv(label_path=label_path)
    
    # Looping over each image in test image file
    for img_name in os.listdir(img_path):
            
        detected_faces = DeepFace.extract_faces(
            img_path=img_path+img_name,
            detector_backend=detector_backend,
        )

        found_faces = DeepFace.find(img_path=img_path+img_name,
                    db_path=label_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=True)


        label_lst = []

        for found_face in found_faces:

            if len(found_face) != 0:
                # Face was detected
                img_label = 'UnKown'
                
                # checking the ArcFace_cosine threshold
                if found_face['ArcFace_cosine'].values[0]> 0.25:
                    img_label_path = found_face['identity'].values[0]
                    img_label = df_label[df_label.image_name == img_label_path[img_label_path.rfind('/') + 1:]]['label'].values[0]
                    
                label_lst.append(img_label)

            else:
                label_lst.append('Unkown')

        img = cv2.imread(img_path+img_name)

        rimg= custom_draw_on(img, detected_faces, label_lst)

        # Output the labeled Image
        cv2.imwrite(f"../report/plots/{img_name}", rimg)

        print(f"Recognitioned Faces in {img_name} are: {label_lst}. Exported image with detected faces at: ../report/plots/{img_name}")

    # To indicate the function terminated successfully
    return True
