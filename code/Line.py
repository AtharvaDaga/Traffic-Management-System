import cv2
import numpy as np

# Video capture setup
cap = cv2.VideoCapture("F:/traffic Management system/Vehicle-Detection-Classification-and-Counting-main/Videos/1.avi")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

# Parameters for counting
min_width_Rect = 80
min_height_Rect = 80
offset = 6  # Allowable error for line crossing detection
horizontal_counter = 0
vertical_counter = 0
detect = []

def center_handle(x, y, w, h):
    """Calculate the center of the vehicle's bounding box."""
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get frame size dynamically
    frame_height, frame_width, _ = frame1.shape
    count_line_position = int(frame_height * 0.75)  # Horizontal line at 75% of frame height
    vertical_line_position = int(frame_width * 0.75)  # Vertical line at 75% of frame width

    # Draw the horizontal and vertical lines
    cv2.line(frame1, (0, count_line_position), (frame_width, count_line_position), (255, 127, 0), 3)
    cv2.line(frame1, (vertical_line_position, 0), (vertical_line_position, frame_height), (0, 255, 255), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_Rect) and (h >= min_height_Rect)
        if not validate_counter:
            continue

        center = center_handle(x, y, w, h)
        detect.append(center)

        # Check if vehicle crosses the horizontal line
        if (center[1] < (count_line_position + offset)) and (center[1] > (count_line_position - offset)):
            horizontal_counter += 1
            cv2.line(frame1, (0, count_line_position), (frame_width, count_line_position), (0, 127, 255), 3)
            if center in detect:
                detect.remove(center)

        # Check if vehicle crosses the vertical line
        if (center[0] < (vertical_line_position + offset)) and (center[0] > (vertical_line_position - offset)):
            vertical_counter += 1
            cv2.line(frame1, (vertical_line_position, 0), (vertical_line_position, frame_height), (0, 127, 255), 3)
            if center in detect:
                detect.remove(center)

    # Display the frame
    cv2.imshow("Vehicle Detection", frame1)

    if cv2.waitKey(1) == 13:  # Press Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()
