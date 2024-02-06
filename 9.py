import cv2
import numpy as np
import pyttsx3

def detect_objects(frame, net, classes, confidence_threshold=0.5):
    height, width, _ = frame.shape

    # Create a blob from the frame and set it as the input to the network
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
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

def draw_objects(frame, objects, classes, frame_center, engine):
    for obj in objects:
        x, y, w, h = obj['box']
        class_id = obj['class_id']
        confidence = obj['confidence']
        
        label = f"{classes[class_id]}: {confidence:.2f}"

        obj_center_x = x + w // 2
        obj_center_y = y + h // 2

        direction = get_direction(frame_center, (obj_center_x, obj_center_y), frame.shape)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Speak the direction
        engine.say(direction)
        engine.runAndWait()

def get_direction(frame_center, obj_center, frame_shape):
    frame_center_x, frame_center_y = frame_center
    obj_center_x, obj_center_y = obj_center

    if obj_center_x < frame_center_x - frame_shape[1] // 4:
        return "Turn Left"
    elif obj_center_x > frame_center_x + frame_shape[1] // 4:
        return "Turn Right"
    else:
        return "Go Straight"

if __name__ == "__main__":
    net = cv2.dnn.readNet("C://Users//Debarghya Kundu//Desktop//blind//darknet//yolov3.weights", "C://Users//Debarghya Kundu//Desktop//blind//darknet//cfg//yolov3.cfg")
    classes = []
    with open("C://Users//Debarghya Kundu//Desktop//blind//darknet//data//coco.names", "r") as f:
        classes = [line.strip() for line in f]

    cap = cv2.VideoCapture(0)

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    while True:
        ret, frame = cap.read()
        frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        objects = detect_objects(frame, net, classes)

        draw_objects(frame, objects, classes, frame_center, engine)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Release the text-to-speech engine
    engine.stop()
