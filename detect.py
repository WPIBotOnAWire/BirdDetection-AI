import cvzone
import cv2

#BIRDS-450-(200 X 200)-99.28.h5
#keras_model.h5
cap = cv2.VideoCapture(0)
myClassifier = cvzone.Classifier('keras_model.h5', 'labels.txt')

while True:
    _, img = cap.read()
    predictions, index = myClassifier.getPrediction(img)
    print(predictions,index)
    cv2.imshow("Image", img)
    cv2.waitKey(1)