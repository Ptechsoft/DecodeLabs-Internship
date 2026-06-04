# Project 4: Object Detection with MobileNet-SSD
# DecodeLabs Internship

import cv2        # OpenCV - loads images, draws boxes, processes visuals
import numpy as np  # NumPy - handles number arrays the model needs

# The 21 object categories this model was trained to detect ──
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Assign a random unique color to each class for its bounding box
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the pre-trained MobileNet-SSD model ─
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",   # the blueprint (architecture)
    "MobileNetSSD_deploy.caffemodel"  # the brain (pre-trained weights)
)
print("[INFO] Model loaded successfully!")


# Load the image from disk
image_path = "test_image.jpg"
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print("[ERROR] Image not found. Check the filename.")
    exit()

# Get the height and width of the image
(h, w) = image.shape[:2]
print(f"[INFO] Image loaded successfully: {w}x{h} pixels")

# Pre-Processing: Convert image to a blob
# This satisfies Gatekeeper Rule #2: Pre-Processing Integrity
print("[INFO] Pre-processing image...")
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),  # resize to network's required size
    0.007843,                        # scale factor (normalizes pixel values)
    (300, 300),                      # target dimensions
    127.5                            # mean subtraction value
)
print("[INFO] Blob constructed successfully!")

# Feed the blob into the network and get predictions
print("[INFO] Running object detection...")
net.setInput(blob)      # feed the pre-processed image into the model
detections = net.forward()  # run the forward pass - this is where AI works!
print(f"[INFO] Detection complete! Analyzing results...")

# Loop over detections and draw bounding boxes
detected_count = 0

for i in range(detections.shape[2]):

    # Extract confidence score for this detection
    confidence = detections[0, 0, i, 2]

    # Gatekeeper Rule #3: Only accept 80% confidence and above
    if confidence > 0.80:
        detected_count += 1

        # Get the class index and label name
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx]

        # Calculate bounding box coordinates on the original image
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box rectangle
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      COLORS[idx], 2)

        # Create label text: "dog: 94.3%"
        text = f"{label}: {confidence * 100:.1f}%"

        # Position text above the box
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # Draw the label text on the image
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        print(f"  Detected: {label} ({confidence * 100:.1f}% confidence)")

#Final Output
print(f"\n[RESULT] {detected_count} object(s) detected above 80% confidence.")

# Save the output image
cv2.imwrite("output_detected.jpg", image)
print("[INFO] Output saved as 'output_detected.jpg'")

# Display the image on screen
cv2.imshow("Project 4 - Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()