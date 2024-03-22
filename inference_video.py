import cv2
from ultralytics import YOLO
import torch


# Set the device to use gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# print if gpu is available
print(f"Using device: {device}")

vid_path = "C:\\Users\\USER\\Documents\\Python Scripts\\fall_detection\\Fall_Detection_Using_Yolov8\\video2.mp4"
model = YOLO('C:\\Users\\USER\\Documents\\Python Scripts\\fall_detection\\Fall_Detection_Using_Yolov8\\fall_det_1.pt')

results_generator = model.predict(
    source=vid_path, # Video file path or RTSP stream URL
    conf=0.8, # Confidence threshold
    iou=0.7, # NMS IoU threshold
    imgsz=640, # Inference size (pixels)
    show=True, # Show results
    save=True, # Save results
    save_txt=False, # Save confidences and locations
    save_conf=True, # Save confidences
    save_crop=False, # Save cropped prediction boxes
    stream=True, # Do not save the video stream
    device=device
)

for results in results_generator:
    # Extract the original image from results
    img = results.orig_img

    # Check if 'pred' attribute exists, which contains bounding box information
    if hasattr(results, 'pred') and hasattr(results.pred[0], 'xyxy'):
        # Render boxes on the image using OpenCV
        for det in results.pred[0].xyxy:
            bbox = det[:4].int().cpu().numpy()
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            confidence = det[4].cpu().numpy()

            # Draw bounding box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # Display confidence
            label = f"Confidence: {confidence:.2f}"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  

    # Display the frame with OpenCV
    cv2.imshow('YOLO Results', img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close OpenCV windows
cv2.destroyAllWindows()
