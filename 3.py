import cv2
import numpy as np

def detect_obstacles(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original frame
    result_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)


    # Determine the direction based on the number of detected contours
    direction = "Forward" if len(contours) == 0 else "Obstacle detected"

    return result_frame, direction

if __name__ == "__main__":
    # Open the default webcam (you can replace 0 with the webcam index if you have multiple cameras)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Perform obstacle detection
        result, direction = detect_obstacles(frame)

        # Display the resulting frame
        cv2.imshow("Obstacle Detection", result)

        # Print the direction
        print("Direction:", direction)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
