

import recognition


label_path = '../data/label'

# Defining models
face_recognition_model = 'ArcFace'
face_detector_model = 'retinaface'

if __name__ == '__main__':

    if task == 'single':
        _ = recognition.single_image_recognition(
                img_path=img_path,
                img_name=img_name,
                label_path=label_path,
                model_name=face_detector_model,
                detector_backend=face_detector_model
            )
    
    elif task == 'multi':
        _ = recognition.multi_image_recognition(
                img_path=img_path,
                label_path=label_path,
                model_name=face_detector_model,
                detector_backend=face_detector_model
            )


