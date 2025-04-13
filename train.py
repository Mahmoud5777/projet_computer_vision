from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(
        resume=True,
        device=0,
        imgsz=640,
        epochs=100
    )

if __name__ == "__main__":
    main()
