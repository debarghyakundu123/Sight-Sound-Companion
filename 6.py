import cv2
import numpy as np

def detect_objects(frame, net, classes, confidence_threshold=0.5):
    height, width, _ = frame.shape

    # Create a blob from the frame and set it as the input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get output from output layer names
    outputs = net.forward(output_layer_names)

    # Process each output
    objects = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                objects.append({
                    'class_id': class_id,
                    'confidence': float(confidence),
                    'box': (x, y, w, h)
                })

    return objects

def draw_objects(frame, objects, classes):
    for obj in objects:
        x, y, w, h = obj['box']
        class_id = obj['class_id']
        label = f"{classes[class_id]}: {obj['confidence']:.2f}"

        color = (0, 255, 0)  # Green color for drawing bounding boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == "__main__":
    # Load YOLO model and classes
    net = cv2.dnn.readNet("C://Users//Debarghya Kundu//Desktop//blind//darknet//yolov3.weights", "C://Users//Debarghya Kundu//Desktop//blind//darknet//cfg//yolov3.cfg")
    classes = []
    with open("C://Users//Debarghya Kundu//Desktop//blind//darknet//data//coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Open the default webcam (you can replace 0 with the webcam index if you have multiple cameras)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect objects in the frame
        objects = detect_objects(frame, net, classes)

        # Draw bounding boxes and labels on the original frame
        draw_objects(frame, objects, classes)

        # Display the resulting frame
        cv2.imshow("Object Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
