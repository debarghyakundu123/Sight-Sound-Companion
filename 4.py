import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np

class ObstacleDetectorApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Obstacle Detector App")

        # Open the video source (webcam)
        self.cap = cv2.VideoCapture(video_source)

        # Create a canvas to display the video feed
        self.canvas = tk.Canvas(root)
        self.canvas.pack()

        # Create a label to display the direction
        self.direction_label = ttk.Label(root, text="Direction: ")
        self.direction_label.pack(pady=10)

        # Create the quit button
        self.quit_button = ttk.Button(root, text="Quit", command=self.quit)
        self.quit_button.pack(pady=10)

        # Start the update loop
        self.update()

    def update(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        # Perform obstacle detection
        result, direction = self.detect_obstacles(frame)

        # Display the resulting frame on the canvas
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(result))
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        # Update the direction label
        self.direction_label.config(text="Direction: " + direction)

        # Schedule the next update
        self.root.after(10, self.update)

    def detect_obstacles(self, frame):
        # Use YOLO for object detection (pre-trained model)
        net = cv2.dnn.readNet("darknet//yolov3.weights", "C:\\Users\\Debarghya Kundu\\Desktop\\blind\\darknet\\cfg\\yolov3.cfg")
        classes = []
        with open("C:\\Users\\Debarghya Kundu\\Desktop\\blind\\darknet\\data\\coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getUnconnectedOutLayersNames()

        # Resize frame and forward it through the network
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)

        # Post-process the detections
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes and labels
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame, "Forward" if len(boxes) == 0 else "Obstacle detected"

    def quit(self):
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObstacleDetectorApp(root)
    root.mainloop()
