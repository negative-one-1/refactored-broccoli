
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# 1. Locate model
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "selfie_segmenter.tflite"


# 2. Configure MediaPipe
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


# 3. Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    segmenter.close()
    raise RuntimeError("Could not access webcam")

print("GokuCam - Background Preservation")
print("Press B to capture empty background")
print("Press Q to exit")


# 4. Background storage
background = None
timestamp = 0

try:
    while True:

        ret, frame = cap.read()

        if not ret:
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

        # Run segmentation
        result = segmenter.segment_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33

        # Extract mask
        confidence_mask = result.confidence_masks[0]

        mask = confidence_mask.numpy_view()

        mask = cv2.resize(
            mask,
            (frame.shape[1], frame.shape[0])
        )

        # Binary mask
        mask = (mask > 0.5).astype(np.uint8)

        # Expand mask to 3 channels
        mask_3ch = np.stack(
            [mask] * 3,
            axis=-1
        )

        # 5. Capture empty background
        if background is None:

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                "Press B when you leave the frame",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        else:

            # 6. Combine person and background
            output = np.where(
                mask_3ch == 1,
                frame,
                background
            )

            display_frame = output

        # Display camera output
        cv2.imshow(
            "GokuCam - Background Preservation",
            display_frame
        )

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):

            if np.mean(mask) > 0.05:
                print("Someone is still in the frame!")
                print("Leave the frame and press B again.")

            else:
                background = frame.copy()
                print("Background captured successfully!")

        elif key == ord('q'):
            break

finally:
    cap.release()
    segmenter.close()
    cv2.destroyAllWindows()

print("GokuCam closed.")
