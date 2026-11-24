
import cv2

# 1. Initialize webcam
cap = cv2.VideoCapture(0)

# Check whether webcam opened successfully
if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("GokuCam started!")
print("Press Q to exit.")

while True:

    # 2. Capture frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # 3. Mirror the webcam feed
    frame = cv2.flip(frame, 1)

    # 4. Display application name
    cv2.putText(
        frame,
        "GokuCam - Live Camera",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # 5. Display webcam feed
    cv2.imshow("GokuCam", frame)

    # 6. Exit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Release webcam resources
cap.release()
cv2.destroyAllWindows()
