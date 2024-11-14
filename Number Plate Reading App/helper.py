import cv2
import numpy as np
from ultralytics import YOLO

# Detection Function
def detect_plate(image, model_path):
    print("[INFO].. Image is loading !")
    image_array = np.asarray(image)
    
    model = YOLO(model_path)
    results = model(image_array)[0]
    
    is_detected = len(results.boxes.data.tolist()) > 0
    cropped_image = None
    plates = []  # List to store detected plates with ID
    
    if is_detected:
        threshold = 0.5
        for idx, result in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if score > threshold:
                cropped_image = image_array[y1:y2, x1:x2]
                cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
                score = score * 100
                class_name = results.names[class_id]
                text = f"{class_name}: %{score:.2f}"
                cv2.putText(image_array, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Store detected plates with unique ID
                plates.append({"id": f"plate_{idx+1}", "plate": text})
                
    else:
        text = "No detection"
        cropped_image = np.zeros((512, 512, 3), np.uint8)
        cv2.putText(image_array, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    
    return image_array, cropped_image, is_detected  # Return 3 values (image, cropped image, detection status)
