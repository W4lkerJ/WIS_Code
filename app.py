import os
import cv2
import numpy as np


def process_image(network, output_layers, classes, image):
    image = cv2.resize(image, None, fx=0.4, fy=0.3)
    height, width, channels = image.shape

    # Convert image into format the nn can process
    blob = cv2.dnn.blobFromImage(image, 0.00392, (220, 220), (0, 0, 0), True, crop=False)

    # Get Prediction
    network.setInput(blob)
    outs = network.forward(output_layers)

    # Map prediction to concrete coordinates
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

                curr_width = int(detection[2] * width)
                curr_height = int(detection[3] * height)

                x = int(center_x - curr_width / 2)
                y = int(center_y - curr_height / 2)

                boxes.append([x, y, curr_width, curr_height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Remove doubled boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    # Draw boxes onto image
    font = cv2.FONT_HERSHEY_PLAIN
    for k in range(len(boxes)):
        if k in indexes:
            x, y, w, h = boxes[k]
            label = str(classes[class_ids[k]])

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y + 30), font, 1, (255, 255, 255), 2)

    return image


def main():
    # Paths & Constants
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(base_dir, "yolo", "yolov3-tiny.weights")
    conf_path = os.path.join(base_dir, "yolo", "yolov3-tiny.cfg")
    class_names_path = os.path.join(base_dir, "yolo", "coco.names")

    # Parameters for camera
    camera_width = 1280
    camera_height = 720
    framerate = 15
    window_width = 1280
    window_height = 720
    flip_method = 2

    # Starting string for RasPi Camera Module V2.1
    start_string = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width={}, height={}, format=NV12, framerate={}/1" \
                   "! nvvidconv flip-method={} ! video/x-raw, width={}, height={}, format=(string)BGRx ! videoconvert" \
                   "! video/x-raw, format=(string)BGR ! appsink".format(camera_width, camera_height, framerate,
                                                                        flip_method, window_width, window_height)

    # Loading the YoloV3 NN
    net = cv2.dnn.readNet(weight_path, conf_path)

    # Loading the class names
    with open(class_names_path, "r") as file:
        classes = [line.strip() for line in file.readlines()]

    # Start the camera and the nn pipeline with OpenCV
    cap = cv2.VideoCapture(start_string, cv2.CAP_GSTREAMER)  # Use 0 if one wants to use the webcam
    if cap.isOpened():
        cv2.namedWindow("RasPi Camera Module V2.1", cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty("RasPi Camera Module V2.1", 0) >= 0:
            ret_val, img = cap.read()

            # Classify objects
            layer_names = net.getLayerNames()
            output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            img = process_image(net, output_layers, classes, img)

            # Resize the image to window size
            img = cv2.resize(img, (window_width, window_height))

            # Show the processed image on the screen
            cv2.imshow("RasPi Camera Module V2.1", img)

            # Stop the program on the ESC key
            key_code = cv2.waitKey(1) & 0xFF
            if key_code == 27:
                break

        # Clearing up
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Error - Could not open camera")


if __name__ == "__main__":
    main()
