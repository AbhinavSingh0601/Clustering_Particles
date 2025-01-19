import cv2
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN


# Function to calculate RMS speed of particles between two frames
def calculate_rms_speed(particle_positions_old, particle_positions_new, frame_time):
    displacements = np.linalg.norm(particle_positions_new - particle_positions_old, axis=1)
    speeds = displacements / frame_time  # Velocity = displacement / time
    rms_speed = np.sqrt(np.mean(speeds**2))  # RMS speed
    return rms_speed


# Function to process frames from folder and apply particle detection, shadow removal, and slow-motion effect
def process_frames_for_slowmotion(input_folder, output_video_path, frame_rate=30, slow_factor=5, target_height=700,
                                  target_width=1080):
    # Get the list of all image files in the folder
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg') or f.endswith('.png')])

    # Set up the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate / slow_factor, (target_width, target_height))

    prev_gray = None
    prev_points = None
    prev_particle_positions = None
    time_ms = 0  # Time in milliseconds
    rms_speeds = []  # To store RMS speeds in m/s
    times = []  # To store times in milliseconds

    # Scaling factor for converting pixels per second to meters per second
    scale_factor = 0.004 / 96  # 4mm diameter = 96 pixels, so scaling factor is 4mm / 96 pixels

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)

        # Get the original height and width of the frame
        height, width, _ = frame.shape

        # Fill the left part (815px) and right part (560px) with black
        frame[:, :815] = [0, 0, 0]  # Fill left part with black
        frame[:, width - 560:] = [0, 0, 0]  # Fill right part with black

        # Resize the frame to target dimensions (700x1080)
        frame_resized = cv2.resize(frame, (target_width, target_height))

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        # Detect particles using thresholding or color segmentation
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Shadow detection: assuming shadows are darker regions near particles
        # Define lower and upper bounds for shadow detection
        shadow_lower = np.array([0, 0, 0], dtype=np.uint8)
        shadow_upper = np.array([80, 80, 80], dtype=np.uint8)

        # Convert to HSV color space to better detect shadows
        hsv = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)
        shadow_mask = cv2.inRange(hsv, shadow_lower, shadow_upper)

        # Set shadow areas to black (exclude from particle detection)
        frame_resized[shadow_mask == 255] = [0, 0, 0]

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_particles = []

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter out small noise particles
                # Calculate particle size (radius)
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                current_particles.append((x, y, radius))
                cv2.circle(frame_resized, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        if prev_gray is not None and prev_points is not None:
            # Calculate optical flow to detect particle velocity
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
            good_new = next_points[status == 1]
            good_old = prev_points[status == 1]

            # Calculate velocity (change in position between frames)
            velocities = np.linalg.norm(good_new - good_old, axis=1)
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                old = tuple(map(int, old))
                new = tuple(map(int, new))

                # Draw the arrowed line representing the velocity vector
                cv2.arrowedLine(frame_resized, old, new, (0, 0, 255), 2)

        # Detect clusters of particles using DBSCAN
        particle_positions = np.array([(x, y) for x, y, _ in current_particles])
        if len(particle_positions) > 0:
            clustering = DBSCAN(eps=30, min_samples=2).fit(particle_positions)
            labels = clustering.labels_

            # Identify clusters and draw bounding boxes around them
            unique_labels = set(labels)
            for label in unique_labels:
                if label != -1:  # -1 represents noise points
                    cluster_points = particle_positions[labels == label]
                    x_min, y_min = np.min(cluster_points, axis=0)
                    x_max, y_max = np.max(cluster_points, axis=0)

                    # Cast these values to integers before drawing the rectangle
                    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                    # Draw the bounding box around the cluster
                    cv2.rectangle(frame_resized, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        # Calculate RMS speed if particles have been detected and previous positions exist
        if prev_particle_positions is not None and len(prev_particle_positions) == len(current_particles):
            prev_positions = np.array([[p[0], p[1]] for p in prev_particle_positions], dtype=np.float32)
            current_positions = np.array([[p[0], p[1]] for p in current_particles], dtype=np.float32)

            # Calculate RMS speed (frame time is based on frame rate, assuming 1 frame = 1/frame_rate seconds)
            rms_speed = calculate_rms_speed(prev_positions, current_positions, 1 / frame_rate)

            # Convert RMS speed to m/s
            rms_speed_mps = rms_speed * scale_factor  # Convert from pixels/s to meters/s

            # Store the time in milliseconds and RMS speed in m/s
            times.append(time_ms)
            rms_speeds.append(rms_speed_mps)

            # Display RMS speed in m/s on the frame
            cv2.putText(frame_resized, f'RMS Speed: {rms_speed_mps:.4f} m/s', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Update previous particle positions for the next frame
        prev_particle_positions = current_particles

        # Increment time by 1/frame_rate (in milliseconds)
        time_ms += 1000 / frame_rate

        # Write the processed frame to the output video
        for _ in range(slow_factor):  # Slow down by repeating frames
            out.write(frame_resized)

        # Show the frame
        cv2.imshow('Particle Tracking', frame_resized)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Create a dataframe with time and RMS speed
    df = pd.DataFrame({'Time (ms)': times, 'RMS Speed (m/s)': rms_speeds})

    # Save the dataframe to a CSV file
    df.to_csv('particle_velocity_data.csv', index=False)

    # Release resources
    out.release()
    cv2.destroyAllWindows()


# Example usage
process_frames_for_slowmotion(r'4mm/16cm_8ml/', 'output_slowmotion.avi', frame_rate=240, slow_factor=100)
