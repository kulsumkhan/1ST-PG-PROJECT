#Description: This program detects faces, and eyes.

# Face detection is performed by using classifiers.
# A classifier is essentially an algorithm that decides whether a given image is positive(face)
# or negative(not a face). We will use the Haar classifier which was named after the Haar wavelets
# because of their similarity. The Haar classifier employs a machine learning algorithm called
# Adaboost for visual object detection

#Resources: https://stackoverflow.com/questions/23720875/how-to-draw-a-rectangle-around-a-region-of-interest-in-python
#Data Camp: https://www.datacamp.com/community/tutorials/face-detection-python-opencv

#import Open CV library
import cv2

#The Haar Classifiers stored as .xml files (Open CV's pretrained classifiers)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Read in the img
img = cv2.imread('dwayne_and_james.jpg') 
faces = face_cascade.detectMultiScale(img, 1.3, 5)

print('Faces found: ', len(faces))


print('The image height, width, and channel: ',img.shape)
print('The coordinates of each face detected: ', faces) 


for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) # Draw rectangle around detected face(s) , #cv2.rectangle(img, pt1, pt2, color, thickness), NOTE: pt1 os upper left and pt2 is bottom right
    roi_face = img[y:y + h, x:x + w] #Get the pixel coordinates within the detected face border the region of interest (ROI), because eyes are located on the face
    eyes = eye_cascade.detectMultiScale(roi_face) #returns the position of the detected eyes
    for (ex, ey, ew, eh) in eyes: #loop over all the coordinates eyes returned and draw rectangles around them using Open CV.
        cv2.rectangle(roi_face, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2) #Draw reactangle around the eyes



font = cv2.FONT_HERSHEY_SIMPLEX
text = cv2.putText(img, 'Face Detected', (0, img.shape[0]), font, 2, (255, 255, 255), 2)

cv2.imshow('imgage',img) 
cv2.waitKey(0)
cv2.destroyAllWindows() 