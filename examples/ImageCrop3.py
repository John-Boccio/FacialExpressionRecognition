import cv2
size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier('C:\\Users\\Owner\\PycharmProjects\\untitled2\\haarcascade_frontalface_default.xml')
#  Above line normalTest
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#Above line test with different calulation
#classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')
#classifier = cv2.CascadeClassifier('lbpcascade_frontalface.xml')


while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, int((im.shape[1] / size, im.shape[0] / size)))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "examples/input" + str(y) + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)

    # Show the image
    cv2.imshow('ImageCrop',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27: #The Esc key
        break