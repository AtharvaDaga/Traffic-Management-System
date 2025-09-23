import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime

# Video capture setup
cap = cv2.VideoCapture("path of the video")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

count_line_position = 400
vertical_line_position = 500
min_width_Rect = 80
min_height_Rect = 80
detect = []
offset = 6  # Allowable Error between pixel
counter = 0

# Define thresholds for traffic density
low_threshold = 15
moderate_threshold = 30

# Initialize Subtractor (optimized version)
algo = cv2.createBackgroundSubtractorKNN(detectShadows=True)

def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def get_traffic_density(count):
    if count < low_threshold:
        return "Low"
    elif count < moderate_threshold:
        return "Moderate"
    else:
        return "High"

# Timer setup
start_time = time.time()
save_interval = 30  # Save every 30 seconds for testing
reset_interval = 30  # Reset every 30 seconds for testing
last_reset_time = time.time()

# Create a DataFrame to store results
results_df = pd.DataFrame(columns=['Date Time', 'Vehicle Count', 'Traffic Density'])

while True:
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Optionally resize frame for processing
    resized_frame = cv2.resize(frame1, (1280, 720))  # Resize for faster processing
    grey = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (5, 5), 3)  # Use a slightly larger kernel for GaussianBlur
    
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((3, 3)), iterations=2)  # Reduce the number of iterations for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Get the width of the current frame
    frame_width = resized_frame.shape[1]

    # Draw the horizontal line across the full width of the frame
    cv2.line(resized_frame, (0, count_line_position), (frame_width, count_line_position), (255, 127, 0), 3)

    # Draw the vertical line 450 pixels from the right
    vertical_line_position_actual = frame_width - vertical_line_position
    cv2.line(resized_frame, (vertical_line_position_actual, 0), (vertical_line_position_actual, resized_frame.shape[0]), (0, 255, 255), 3)

    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_Rect) and (h >= min_height_Rect)
        if not validate_counter:
            continue

        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(resized_frame, "Vehicle NO : " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(resized_frame, center, 4, (0, 0, 255), -1)

        for (cx, cy) in detect:
            if (cy < (count_line_position + offset)) and (cy > (count_line_position - offset)):
                counter += 1
                cv2.line(resized_frame, (0, count_line_position), (frame_width, count_line_position), (0, 127, 255), 3)
                detect.remove((cx, cy))
                print("VEHICLE COUNTER : " + str(counter))
            elif (cx < (vertical_line_position_actual + offset)) and (cx > (vertical_line_position_actual - offset)):
                counter += 1
                cv2.line(resized_frame, (vertical_line_position_actual, 0), (vertical_line_position_actual, resized_frame.shape[0]), (127, 0, 255), 3)
                detect.remove((cx, cy))
                print("VEHICLE COUNTER : " + str(counter))

    traffic_density = get_traffic_density(counter)
    cv2.putText(resized_frame, f"Traffic Density: {traffic_density}", (450, 120), cv2.FONT_ITALIC, 2, (0, 255, 0), 5)
    cv2.putText(resized_frame, "VEHICLE COUNTER : " + str(counter), (450, 70), cv2.FONT_ITALIC, 2, (0, 0, 255), 5)

    cv2.imshow("OUTPUT VIDEO", resized_frame)

    # Check if save_interval has passed
    if time.time() - start_time >= save_interval:
        start_time = time.time()  # Reset the save timer
        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")

        # Create new data as a DataFrame
        new_data = pd.DataFrame({'Date Time': [date_time], 'Vehicle Count': [counter], 'Traffic Density': [traffic_density]})

        # Append new data to results_df
        results_df = pd.concat([results_df, new_data], ignore_index=True)

        # Save DataFrame to CSV
        results_df.to_csv('F:/traffic Management system/Final/Counts/7.csv', index=False)
        print("6.csv")

    # Reset the counter every reset_interval
    if time.time() - last_reset_time >= reset_interval:
        last_reset_time = time.time()
        counter = 0  # Reset the counter

    if cv2.waitKey(1) == 13:  # Press Enter key to exit
        break

cv2.destroyAllWindows()
cap.release()
