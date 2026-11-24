
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "selfie_segmenter.tflite"

PARTICLE_COUNT = 1000
AURA_DURATION = 35
DISAPPEAR_DURATION = 65
REAPPEAR_DURATION = 55


# ==========================================
# 2. INITIALIZE MEDIAPIPE
# ==========================================

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


# ==========================================
# 3. INITIALIZE WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    segmenter.close()
    raise RuntimeError("Could not access webcam")


print("GokuCam Teleportation Engine Started!")
print("B = Capture background")
print("SPACE = Teleport")
print("R = Reappear")
print("Q = Quit")


# ==========================================
# 4. INITIALIZE VARIABLES
# ==========================================

background = None

state = "NORMAL"

timestamp = 0
animation_frame = 0

saved_person = None
saved_mask = None

particles = None
velocities = None
target_positions = None

rng = np.random.default_rng()


# ==========================================
# 5. IMAGE COMPOSITING
# ==========================================

def composite_person(frame, mask, background):

    # Ensure mask is 2D
    mask = np.squeeze(mask)

    # Expand mask for RGB broadcasting
    mask_3ch = mask[:, :, np.newaxis]

    return np.where(
        mask_3ch,
        frame,
        background
    ).astype(np.uint8)

    mask_3ch = np.repeat(
        mask[:, :, np.newaxis],
        3,
        axis=2
    )

    return np.where(
        mask_3ch,
        frame,
        background
    ).astype(np.uint8)


# ==========================================
# 6. CREATE PARTICLES
# ==========================================

def create_particles(mask):

    global particles, velocities, target_positions

    # Find all pixels belonging to the person
    points = np.argwhere(mask > 0)

    if len(points) == 0:
        return False

    # Select random points from the silhouette
    count = min(PARTICLE_COUNT, len(points))

    selected = rng.choice(
        len(points),
        size=count,
        replace=False
    )

    selected_points = points[selected]

    # Convert (y, x) to (x, y)
    particles = selected_points[:, ::-1].astype(float)

    target_positions = particles.copy()

    # Calculate body center
    center = particles.mean(axis=0)

    # Calculate direction away from body center
    directions = particles - center

    # Normalize directions
    lengths = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True
    )

    directions = directions / np.maximum(lengths, 1)

    # Add random movement
    random_movement = rng.normal(
        0,
        1.5,
        size=particles.shape
    )

    directions += random_movement

    # Normalize again
    lengths = np.linalg.norm(
        directions,
        axis=1,
        keepdims=True
    )

    directions = directions / np.maximum(lengths, 1)

    # Assign particle speeds
    speeds = rng.uniform(
        1.5,
        6.0,
        size=(count, 1)
    )

    velocities = directions * speeds

    return True


# ==========================================
# 7. DRAW ENERGY AURA
# ==========================================

def apply_aura(frame, mask):

    # Convert mask into an image
    mask_image = mask.astype(np.uint8) * 255

    # Expand the silhouette
    kernel = np.ones((15, 15), np.uint8)

    expanded = cv2.dilate(
        mask_image,
        kernel,
        iterations=2
    )

    # Create soft glowing edges
    glow = cv2.GaussianBlur(
        expanded,
        (0, 0),
        12
    )

    # Blue energy color (BGR format)
    aura_color = np.zeros_like(frame)

    aura_color[:, :, 0] = 255
    aura_color[:, :, 1] = 110
    aura_color[:, :, 2] = 35

    # Convert glow to transparency
    alpha = (
        glow.astype(np.float32) / 255.0
    ) * 0.65

    alpha = alpha[:, :, np.newaxis]

    result = (
        frame.astype(np.float32) * (1 - alpha)
        + aura_color.astype(np.float32) * alpha
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ==========================================
# 8. DRAW PARTICLES
# ==========================================

def draw_particles(frame, positions, progress):

    height, width = frame.shape[:2]

    # Particles gradually become smaller
    radius = max(
        1,
        int(4 * (1 - progress) + 1)
    )

    # Particles gradually become dimmer
    brightness = int(
        255 * (1 - progress)
    )

    color = (
        brightness,
        int(brightness * 0.8),
        int(brightness * 0.35)
    )

    for x, y in positions:

        x = int(x)
        y = int(y)

        if 0 <= x < width and 0 <= y < height:

            cv2.circle(
                frame,
                (x, y),
                radius,
                color,
                -1,
                cv2.LINE_AA
            )

    return frame


# ==========================================
# 9. MAIN APPLICATION LOOP
# ==========================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert frame to RGB
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

        # Extract person mask
        confidence = result.confidence_masks[0].numpy_view()

        # Remove unnecessary singleton dimensions
        confidence = np.squeeze(confidence)

        mask = (
            confidence > 0.5
        ).astype(np.uint8)

        # ----------------------------------
        # NORMAL STATE
        # ----------------------------------

        if state == "NORMAL":

            if background is not None:

                output = composite_person(
                    frame,
                    mask,
                    background
                )

            else:

                output = frame.copy()

                cv2.putText(
                    output,
                    "Press B to capture background",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        # ----------------------------------
        # AURA STATE
        # ----------------------------------

        elif state == "AURA":

            base = composite_person(
                saved_person,
                saved_mask,
                background
            )

            output = apply_aura(
                base,
                saved_mask
            )

            animation_frame += 1

            if animation_frame >= AURA_DURATION:

                state = "DISAPPEARING"
                animation_frame = 0

        # ----------------------------------
        # DISAPPEARING STATE
        # ----------------------------------

        elif state == "DISAPPEARING":

            output = background.copy()

            # Move particles outward
            particles += velocities

            # Gradually slow particle movement
            velocities *= 0.985

            progress = (
                animation_frame / DISAPPEAR_DURATION
            )

            output = draw_particles(
                output,
                particles,
                progress
            )

            animation_frame += 1

            if animation_frame >= DISAPPEAR_DURATION:

                state = "VANISHED"
                animation_frame = 0

        # ----------------------------------
        # VANISHED STATE
        # ----------------------------------

        elif state == "VANISHED":

            output = background.copy()

            cv2.putText(
                output,
                "TELEPORTATION COMPLETE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 180, 50),
                2
            )

            cv2.putText(
                output,
                "Press R to reappear",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # ----------------------------------
        # REAPPEARING STATE
        # ----------------------------------

        elif state == "REAPPEARING":

            output = background.copy()

            progress = min(
                animation_frame / REAPPEAR_DURATION,
                1.0
            )

            # Smooth particle convergence
            smooth_progress = (
                progress * progress *
                (3 - 2 * progress)
            )

            current_positions = (
                particles * (1 - smooth_progress)
                + target_positions * smooth_progress
            )

            output = draw_particles(
                output,
                current_positions,
                1 - progress
            )

            animation_frame += 1

            if animation_frame >= REAPPEAR_DURATION:

                state = "NORMAL"
                animation_frame = 0

        # ----------------------------------
        # DISPLAY OUTPUT
        # ----------------------------------

        cv2.putText(
            output,
            f"State: {state}",
            (20, output.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "GokuCam - Teleportation",
            output
        )

        # ----------------------------------
        # KEYBOARD CONTROLS
        # ----------------------------------

        key = cv2.waitKey(1) & 0xFF

        # Capture empty background
        if key == ord('b') or key == ord('B'):

            person_ratio = np.mean(mask)

            print(f"Person detection ratio: {person_ratio:.4f}")

            if person_ratio < 0.02:

                background = frame.copy()

                print("Background captured successfully!")

            else:

                print("Person detected! Please leave the camera frame first.")

        # Start teleportation
        elif key == 32:

            if state == "NORMAL" and background is not None:

                if np.mean(mask) > 0.02:

                    saved_person = frame.copy()
                    saved_mask = mask.copy()

                    if create_particles(saved_mask):

                        state = "AURA"
                        animation_frame = 0

                        print("Goku teleportation activated!")

                    else:

                        print("No person detected!")

                else:

                    print("No person detected in the frame.")

            elif background is None:

                print("Capture background first using B.")

        # Reappear
        elif key == ord('r'):

            if state == "VANISHED":

                state = "REAPPEARING"
                animation_frame = 0

                print("Goku reappearance activated!")

        # Exit
        elif key == ord('q'):
            break

finally:

    cap.release()
    segmenter.close()
    cv2.destroyAllWindows()

print("GokuCam closed successfully!")
