# coin_detector.py
import cv2
from ultralytics import YOLO

# ========================
# 1. Configuration
# ========================
DATASET_YAML = "data.yaml"  
MODEL_WEIGHTS = "yolov8s.pt"  
CURRENCY_VALUES = {
    0: 50.00,  # 50 euro
    1: 20.00,  # 20 euro
    2: 10.00,  # 10 euro
    3: 5.00,   # 5 euro
    4: 2.00,   # 2 euro
    5: 1.00,   # 1 euro
    6: 0.50,   # 50 cent
    7: 0.20,   # 20 cent
    8: 0.10,   # 10 cent
    9: 0.05,   # 5 cent
    10: 0.02,  # 2 cent
    11: 0.01   # 1 cent
}

# ========================
# 2. Validation Function
# ========================
def validate_model(model):
    metrics = model.val()
    print(f"Validation Results:")
    print(f" - mAP50: {metrics.box.map50:.2f}")
    print(f" - mAP50-95: {metrics.box.map:.2f}")

# ========================
# 4. Prediction Function
# ========================
import cv2

def predict_image(model, image_path):
    results = model.predict(image_path, save=False,conf=0.3) 
    total = 0.0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            total += CURRENCY_VALUES.get(class_id, 0.0)

    print(f"Total amount: EURO{total:.2f}")
    annotated_img = results[0].plot()
    cv2.putText(
        annotated_img,
        f"Total:  {total:.2f}  EURO",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,        
        (0, 255, 0),
        3          
    )
    cv2.imshow("Prediction", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ========================
# 6. Main Program
# ========================
if __name__ == "__main__":
    trained_model = YOLO("runs/detect/train/weights/best.pt")
    validate_model(trained_model)
    predict_image(trained_model, "test_img.jpg")
    trained_model.export(format="onnx")