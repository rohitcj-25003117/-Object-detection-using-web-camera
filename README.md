WORKSHOP 2 – Object Detection Using Web Camera

Aim

To perform real-time object detection using a trained deep-learning object detection model through the laptop camera using OpenCV.

Software Required

Anaconda – Python 3.7

Jupyter Notebook / VS Code

OpenCV (cv2)

NumPy

MobileNet-SSD Model

Caffe Model Files

Algorithm

Step 1:

Import the required libraries such as OpenCV and NumPy.

Step 2:

Load the trained MobileNet-SSD model using the configuration file and trained model weights.

Step 3:

Define the object classes supported by the trained model.

Step 4:

Initialize the laptop webcam using cv2.VideoCapture(0).

Step 5:

Capture video frames continuously from the webcam.

Step 6:

Resize each frame and convert it into a blob using cv2.dnn.blobFromImage().

Step 7:

Pass the blob through the trained neural network using net.forward().

Step 8:

Check the confidence score of each detected object.

Step 9:

Obtain the class name and bounding-box coordinates for detections above the confidence threshold.

Step 10:

Draw bounding boxes around detected objects using cv2.rectangle().

Step 11:

Display the object name and confidence percentage using cv2.putText().

Step 12:

Display the detected video in real time.

Step 13:

Press S to save the current detected frame as an image.

Step 14:

Press Q to stop the webcam.

Step 15:

Release the webcam and close all OpenCV windows.

Developed By

Name: CJ ROHIT

Register No: 212224243005

Program

import cv2
import numpy as np

# Load MobileNet-SSD Model
model_path = r"C:\Users\CJ ROHIT\mobilenet_iter_73000.caffemodel"
config_path = r"C:\Users\CJ ROHIT\deploy.prototxt"

net = cv2.dnn.readNet(model_path, config_path)

print("Model loaded successfully.")

# Object Classes
classes = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Start Laptop Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam could not be opened.")
else:
    print("Webcam started successfully.")
    print("Press S = Save detected output")
    print("Press Q = Stop webcam")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read webcam frame.")
            break

        height, width = frame.shape[:2]

        # Create Blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )

        # Object Detection
        net.setInput(blob)
        detections = net.forward()

        # Process Detected Objects
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > 0.50:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )

                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width - 1, x2)
                y2 = min(height - 1, y2)

                label = classes[class_id]

                text = "{}: {:.2f}%".format(
                    label, confidence * 100
                )

                # Draw Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display Object Name
                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Display Real-Time Output
        cv2.imshow(
            "WORKSHOP 2 - Real-Time Object Detection",
            frame
        )

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF

        # Press S to Save Output
        if key == ord("s"):

            output_path = r"C:\Users\CJ ROHIT\workshop2_output.jpg"

            cv2.imwrite(output_path, frame)

            print("Output saved successfully!")
            print(output_path)

        # Press Q to Stop
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Object detection stopped successfully.")

Output

Real-Time Object Detection

The laptop webcam captures live video and the trained model detects objects in real time.

Output:

Paste your real-time object detection screenshot here.

<img width="608" height="173" alt="image" src="https://github.com/user-attachments/assets/73c01f33-7f5e-4551-a5f9-9643f06e7e44" />


Detected Object with Bounding Box

The detected object is displayed with a bounding box, object name, and confidence percentage.

Output:

Paste your saved workshop2_output.jpg screenshot here.

<img width="214" height="161" alt="image" src="https://github.com/user-attachments/assets/2f707704-c5ad-4e89-bbe6-7d198e806f49" />


Display Saved Output in Jupyter Notebook

After pressing S, create a new Jupyter cell and run:

from IPython.display import display, Image

display(
    Image(
        filename=r"C:\Users\CJ ROHIT\workshop2_output.jpg"
    )
)

Output:

Paste the output screenshot from the Jupyter cell here.

<img width="492" height="379" alt="image" src="https://github.com/user-attachments/assets/d665283f-3f56-4338-93c2-855251f632a6" />

Result

Thus, real-time object detection was successfully performed using the laptop webcam and a trained MobileNet-SSD model with OpenCV. The detected objects were displayed with bounding boxes, object labels, and confidence scores.
