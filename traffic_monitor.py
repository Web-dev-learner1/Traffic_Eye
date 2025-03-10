import cv2
from tracker import EuclideanDistTracker

# Initialize the tracker
tracker = EuclideanDistTracker()

# Load the video file
cap = cv2.VideoCapture("D:/C_Vision/Traffic_Eye/traffic_1.mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Initialize background subtractor for object detection
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Traffic light states
RED = (0, 0, 255)    # Red
GREEN = (0, 255, 0)  # Green

# Traffic light timer
traffic_light_state = GREEN
state_change_counter = 0  # Counter to simulate timer for traffic lights

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the ROI (Region of Interest)
    height, width, _ = frame.shape
    roi = frame[200:432, 500:768]

    # Apply background subtraction to the ROI
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    # Filter contours by area and draw bounding boxes
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700:  # Minimum area threshold to consider an object
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, w, h])

    # Update tracker with detected bounding boxes
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        x, y, w, h, obj_id = box_id
        cv2.putText(roi, str(obj_id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Control traffic light based on unique object count
    unique_count = len(set([box_id[4] for box_id in boxes_ids]))
    if unique_count > 3:
        traffic_light_state = RED  # More than 3 objects detected, traffic light is red
        state_change_counter = 30  # Set timer for state change (simulate duration)
    elif unique_count == 0:
        traffic_light_state = GREEN  # No objects, traffic light is green
        state_change_counter = 0  # Reset the timer

    # Draw the traffic light state on the screen
    traffic_light_position = (width - 100, 50)  # Adjust position to be visible
    cv2.rectangle(frame, (traffic_light_position[0] - 40, traffic_light_position[1] - 40),
                  (traffic_light_position[0] + 40, traffic_light_position[1] + 40),
                  (255, 255, 255), -1)  # White background for the traffic light
    cv2.circle(frame, traffic_light_position, 30, (0, 0, 0), -1)  # Outline of the indicator
    cv2.circle(frame, traffic_light_position, 25, traffic_light_state, -1)  # Filled circle based on state

    # Display the traffic light state text on the screen
    cv2.putText(frame, f"Traffic Light: {'Green' if traffic_light_state == GREEN else 'Red'}",
                (traffic_light_position[0] - 150, traffic_light_position[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display ROI and the entire frame
    cv2.imshow("ROI", roi)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Exit the loop when 'ESC' key is pressed
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
