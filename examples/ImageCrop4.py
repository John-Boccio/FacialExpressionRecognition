import cv2
import os

def facechop(image):
    facedata = 'C:\\Users\\Owner\\PycharmProjects\\untitled2\\haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(facedata)
    DIR = 'C:/Users/Owner/PycharmProjects/untitled2/input/'
    pictures = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]

    for pic in pictures:
        img = cv2.imread(DIR + str(pic))

        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

            sub_face = img[y:y+h, x:x+w]
            face_file_name = "C:/Users/Owner/PycharmProjects/untitled2/cropmethod4/" + str(y) + ".jpg"
            cv2.imwrite(face_file_name, sub_face)

        cv2.imshow(image, img)

if __name__ :

    facechop("image")

    while(True):
        key = cv2.waitKey(30)
        if key in [27, ord('Q'), ord('q')]:
            break