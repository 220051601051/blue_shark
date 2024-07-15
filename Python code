import cv2
import time
import numpy as np

# Function to detect vehicles using improved contour detection
def detect_vehicle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 1000  # Adjust based on your needs
    vehicle_detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            # Draw a bounding box around the detected vehicle
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            vehicle_detected = True

    return vehicle_detected

# Function to estimate distance to detected vehicle (simulated)
def estimate_distance():
    # Simulated distance estimation (random distance between 5 and 50 meters)
    return 5 + np.random.rand() * 45

# Function to calculate time-to-collision (TTC) based on distance and relative speed
def calculate_ttc(distance, relative_speed):
    if relative_speed <= 0:
        return float('inf')
    return distance / relative_speed

# Function to alert the driver (for demonstration, prints to console)
def alert_driver(message):
    print(f"ALERT: {message}")

# Main function to run collision avoidance system with video input
def collision_avoidance_with_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    relative_speed = 10  # Example relative speed in m/s
    ttc_threshold = 0.6  # Example TTC threshold in seconds

    target_width = 640
    target_height = 480

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (target_width, target_height))

        vehicle_detected = detect_vehicle(frame)

        if vehicle_detected:
            distance = estimate_distance()
            ttc = calculate_ttc(distance, relative_speed)

            if ttc <= ttc_threshold:
                alert_driver(f"Collision imminent! Time to collision: {ttc:.2f} seconds.")
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            else:
                print(f"No immediate collision threat. TTC: {ttc:.2f} seconds.")

        else:
            print("No vehicle detected.")

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

    cap.release()
    cv2.destroyAllWindows()

# Example usage: Provide path to your video file
video_path = 'v.mp4'
collision_avoidance_with_video(video_path)
