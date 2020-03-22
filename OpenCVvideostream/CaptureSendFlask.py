# File to capture images from Camera and send to Flask server
# import the necessary packages
from OpenCVvideostream.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()


# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def detect_faces():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # frame=crops
        frame = vs.read()
        face_data = "./image_processing/resources/haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(face_data)

        opencv_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        minisize = (opencv_img.shape[1], opencv_img.shape[0])
        miniframe = cv2.resize(opencv_img, minisize)

        faces = cascade.detectMultiScale(miniframe)
        crops = []
        for f in faces:
            x, y, w, h = [v for v in f]
            sub_face = opencv_img[y:y + h, x:x + h]
            sub_face = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
            crop = {
                "coord": (x, y),
                "size": (w, h),
                "img": sub_face
            }
            crops.append(crop)
        # frame = imutils.resize(frame, width=400)

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(crops, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, crops.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        # if total > frameCount:
            # detect faces in the image
            #  faces= cascade.detectMultiScale(miniframe)

            # check to see if face was found in the frame
            if faces is not None:
                # unpack the tuple and draw the box surrounding the
                # "face area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = faces
                cv2.rectangle(crops, (minX, minY), (maxX, maxY),
                              (0, 0, 255), 2)

        # update the background model and increment the total number
        # of frames read thus far
        # md.update(gray)
        total += 1

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = crops.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32,
                    help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    # start a thread that will perform face detection
    t = threading.Thread(target=detect_faces, args=(
        args["frame_count"],))
    t.daemon = True
    t.start()

# release the video stream pointer
vs.stop()
