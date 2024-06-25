import cv2
import numpy as np

def draw_bounding_box(image_path: str):
    img = cv2.imread(image_path.replace("\\", "/"))

    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY)[1]

    # get contours
    result = img.copy()
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    #x,y,w,h = 0,0,0,0
    for cntr in contours:
        x,y,w,h = cv2.boundingRect(cntr)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
    #(x,y,w,h)
        return ([x,y,w,h], [x,y,x+w,y+h], result)
    # save resulting image
    #cv2.imwrite('zebrafish_groundtruth.jpg',result)      

