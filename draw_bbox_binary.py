import cv2
import numpy as np

def draw_bounding_box(image_path: str):
    img = cv2.imread(image_path.replace("\\", "/"))
    #print(img)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold (assuming the input image is in the range 0 to 1)
    thresh = cv2.threshold(gray, 0.5, 1.0, cv2.THRESH_BINARY)[1]

    # Get contours
    result = img.copy()
    contours = cv2.findContours(np.uint8(thresh), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Change color to red (BGR format)

    return ([x, y, w, h], [x, y, x+w, y+h], result)
