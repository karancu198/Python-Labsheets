
import cv2
import numpy as np

# Q1. Load and display an image
img = cv2.imread("image.jpg")
if img is not None:
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not found")


# Q2. Convert color image to grayscale
img = cv2.imread("image.jpg")
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Color Image", img)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q3. Display RGB and HSV versions
img = cv2.imread("image.jpg")
if img is not None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("RGB Image", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imshow("HSV Image", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q4. Resize image to 200x200
img = cv2.imread("image.jpg")
if img is not None:
    resized = cv2.resize(img, (200, 200))
    cv2.imshow("Resized Image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q5. Rotate image by 45 degrees
img = cv2.imread("image.jpg")
if img is not None:
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), 45, 1)
    rotated = cv2.warpAffine(img, matrix, (w, h))
    cv2.imshow("Rotated Image", rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q6. Image flipping
img = cv2.imread("image.jpg")
if img is not None:
    horizontal = cv2.flip(img, 1)
    vertical = cv2.flip(img, 0)
    both = cv2.flip(img, -1)
    cv2.imshow("Horizontal Flip", horizontal)
    cv2.imshow("Vertical Flip", vertical)
    cv2.imshow("Both Flip", both)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q7. Crop a specific region
img = cv2.imread("image.jpg")
if img is not None:
    cropped = img[50:250, 50:250]
    cv2.imshow("Cropped Image", cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q8. Detect faces using Haar Cascade
img = cv2.imread("image.jpg")
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q9. Detect eyes using Haar Cascade
img = cv2.imread("image.jpg")
if img is not None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_eye.xml"
    )
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h),
                      (255, 0, 0), 2)

    cv2.imshow("Eye Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Q10. Identify geometric shapes
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
cv2.rectangle(img, (200, 50), (300, 150), (255, 255, 255), -1)
pts = np.array([[100, 250], [50, 350], [150, 350]])
cv2.fillPoly(img, [pts], (255, 255, 255))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(
    gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for contour in contours:
    approx = cv2.approxPolyDP(
        contour, 0.04 * cv2.arcLength(contour, True), True
    )
    sides = len(approx)

    if sides == 3:
        shape = "Triangle"
    elif sides == 4:
        shape = "Rectangle"
    else:
        shape = "Circle"

    x, y, w, h = cv2.boundingRect(approx)
    cv2.putText(img, shape, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

cv2.imshow("Shape Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Q11. Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Q12. Convert video frames to grayscale
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Video", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# Q13. Capture and save 100 webcam frames
cap = cv2.VideoCapture(0)
count = 0

while count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Frames", frame)
    cv2.imwrite(f"frame_{count+1}.jpg", frame)
    count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Frames saved:", count)


# Q14. Draw rectangle, circle and line
canvas = np.zeros((400, 400, 3), dtype=np.uint8)

cv2.rectangle(canvas, (50, 50), (200, 200),
              (0, 255, 0), 3)
cv2.circle(canvas, (300, 100), 50,
           (255, 0, 0), 3)
cv2.line(canvas, (50, 300), (350, 300),
         (0, 0, 255), 3)

cv2.imshow("Drawing Shapes", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Q15. Add text with different fonts and sizes
img = np.zeros((400, 600, 3), dtype=np.uint8)

cv2.putText(img, "OpenCV Python", (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2)

cv2.putText(img, "Computer Vision", (50, 160),
            cv2.FONT_HERSHEY_COMPLEX, 1.2,
            (255, 0, 0), 2)

cv2.putText(img, "Image Processing", (50, 240),
            cv2.FONT_HERSHEY_TRIPLEX, 0.8,
            (0, 0, 255), 2)

cv2.imshow("Text Styles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
