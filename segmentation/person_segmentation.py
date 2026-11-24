
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# 1. Locate our trained segmentation model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "selfie_segmenter.tflite"


# 2. Check if model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Segmentation model not found: {MODEL_PATH}"
    )


# 3. Initialize MediaPipe Image Segmenter
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode

options = ImageSegmenterOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    output_confidence_masks=True
)

segmenter = ImageSegmenter.create_from_options(options)


# 4. Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    segmenter.close()
    raise RuntimeError("Could not access webcam")


print("GokuCam segmentation started!")
print("Press Q to exit.")


# 5. Main processing loop
frame_timestamp_ms = 0

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Process video frame
        result = segmenter.segment_for_video(
            mp_image,
            frame_timestamp_ms
        )

        frame_timestamp_ms += 33

        # Extract confidence mask
        confidence_mask = result.confidence_masks[0]

        mask = confidence_mask.numpy_view()

        # Resize mask to original frame dimensions
        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0])
        )

        # Apply threshold
        binary_mask = (mask > 0.2).astype(np.uint8) * 255

        # Extract person
        foreground = cv2.bitwise_and(
            frame,
            frame,
            mask=binary_mask
        )

        # Display results
        cv2.imshow("Original Webcam", frame)

        cv2.imshow("Segmentation Mask", binary_mask)

        cv2.imshow("Extracted Person", foreground)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    segmenter.close()
    cv2.destroyAllWindows()

print("GokuCam closed successfully!")
