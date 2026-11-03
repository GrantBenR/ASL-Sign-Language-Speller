import ultralytics
import os
import cv2

def TrainModel(
        base_model: str = "yolov8n.pt",
        dataset_dir: str = "dataset",
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        training_device: str = "0"
    ) -> ultralytics.YOLO:
    print(f"BASE MODEL: {base_model}")
    model = ultralytics.YOLO(base_model)

    data_yaml_path = os.path.join(os.getcwd(), dataset_dir, "data.yaml")
    print(f"TRAINING OFF DATA AT: {data_yaml_path}")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=training_device,
    )

    print("TRAINING COMPLETE!")

    # ==========================================
    # 3. USE THE MODEL FOR INFERENCE
    # ==========================================
    # Ultralytics saves the best weights during training usually at runs/detect/train/weights/best.pt
    best_model_path = os.path.join(os.getcwd(), dataset_dir, "runs", "detect", "train", "weights", "best.pt")

    # Load your newly trained custom model
    trained_model = ultralytics.YOLO(best_model_path)

    return trained_model

def TestModel(
        model_to_test: ultralytics.YOLO,
        test_image_path = "path/to/your/test_image.jpg"
    ) -> int:
    if os.path.exists(test_image_path):
        inference_results = model_to_test.predict(
            source=test_image_path, 
            conf=0.25, 
            save=True
        )
        if inference_results:
            print(f"Inference complete! Results saved in runs/detect/predict")
            return 0
        else:
            return -1
    else:
        print(
            f"Test image not found at {test_image_path}. Skipping inference test script."
        )
        return -1

def CaptureImage(
        camera_device_id: int = 0,
        capture_path: str = "assets/images/captured_frame.jpg"
    ) -> int:
    """Capture image from device

    Args:
        camera_device_id (int, optional): camera device id. Defaults to 0.
        capture_path (str, optional): path to save capture to. Defaults to "captured_frame.jpg".

    Returns:
        int: return value
    """
    try:
        return_code = 0
        camera = cv2.VideoCapture(0)

        success, frame = camera.read()

        if success:
            print(capture_path)
            cv2.imwrite(capture_path, frame)
        else:
            return_code = -1
        camera.release()

        cv2.destroyAllWindows()
        return return_code
    except Exception as e:
        return -1

def DetectObjects(
        input_image_path: str = "assets/images/captured_frame.jpg",
        output_image_path: str = "assets/images/annotated_image.jpg",
        yolo_model: str = "yolov8n.pt"
    ) -> list[dict]:
    """Detect objects in image and return the image with bounding boxes

    Args:
        input_image_path (str, optional): path of image to detect. Defaults to "assets/images/captured_frame.jpg".
        output_image_path (str, optional): path to output bounding box image to. Defaults to "assets/images/annotated_image.jpg".
        yolo_model (str, optional): model to use for detection. Defaults to "yolov8n.pt".

    Returns:
        str: path of bb image
    """
    try:
        model = ultralytics.YOLO(
            model=yolo_model
        )
        results = model(
            source=input_image_path
        )
        
        result = results[0]

        result.save(filename=output_image_path)
        print(output_image_path)
        return ""
    except Exception as e:
        print(e)
        return ""

def main():
    # trained_model = TrainModel(
    #     epochs=50,
    #     training_device="0"
    # )
    if not os.path.exists("C:/Users/test/Grant/Projects/WSU/CS4900/CS4900-Mini-Project-3/dataset/runs/detect/train-4/weights/best.pt"):
        trained_model = TrainModel()
    else:
        capture_path = os.path.join(os.getcwd(), "wumbo.jpg")
        detected_path = os.path.join(os.getcwd(), "wumbo_detected.jpg")
        model_path = os.path.join(os.getcwd(), "dataset", "runs", "detect", "train-4", "weights", "best.pt")
        # CaptureImage(capture_path=capture_path)
        DetectObjects(
            input_image_path=capture_path, 
            output_image_path=detected_path, 
            yolo_model=model_path
        )
        model = ultralytics.YOLO(model_path)
        results = model.predict(
            source=0, conf=0.15, show=True  # Opens a live window of your webcam feed
        )

if __name__ == "__main__":
    main()