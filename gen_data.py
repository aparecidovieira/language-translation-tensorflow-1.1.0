import cv2
import numpy as np


MIN_AREA_CONTOUR = 0
MAX_AREA_CONTOUR = 400


RESIDED_WIDTH = 10
RESIDED_HEIGHT = 10

class_path = 'classification.txt'
flat_images_path = 'flat_images.txt'
train_image_path = 'training_ocr_extended_a.png'
train1 = 'training_ocr_extended_a.png'
image = 'training_ocr_extended_a.png'



def generate_data(img):
    
    imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, imgThresh = cv2.threshold(imgGrayScale, 127, 255, cv2.THRESH_BINARY_INV)
    _, countours, _ = cv2.findContours(imgThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('Thresh', imgThresh)
    
    flattenedImages = np.empty((0, RESIDED_WIDTH*RESIDED_HEIGHT))
    classification = []
    for countour in countours:
    	x, y, w, h = cv2.boundingRect(countour)
    	if MIN_AREA_CONTOUR < cv2.contourArea(countour) < MAX_AREA_CONTOUR and h > 2:
    		#draw rectangle around area detected
    		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    		imgResized = cv2.resize(imgThresh[y:y+h, x:x+w], (RESIDED_WIDTH, RESIDED_HEIGHT))
    		cv2.imshow('Resized ROI', imgResized)
    		#input from user
    		key = cv2.waitKey(0)
    		if key == 27:
    			exit()

    		classification.append(key)
    		imgFlat = imgResized.reshape((1, RESIDED_WIDTH*RESIDED_HEIGHT))
    		flattenedImages = np.append(flattenedImages, imgFlat, 0)



    classification = np.array(classification, np.float32)
    final_class = classification.reshape((classification.size, 1))
    class_file = open(class_path, 'ab')
    fla_file = open(flat_images_path, 'ab')
    np.savetxt(class_file, final_class)
    np.savetxt(fla_file, flattenedImages)


img = cv2.imread(image)
if img is None:
    print('Image loading have failed')
    exit(1)

		
generate_data(img)
cv2.imshow('Image', img)


cv2.destroyAllWindows()
       
    



