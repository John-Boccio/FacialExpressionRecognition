from imutils.video import VideoStream
from flask import Flask, Response, render_template
import threading
import datetime
import imutils
import cv2
from image_processing import histogram_equalization
from image_processing import crop_faces
from image_processing import crop_face_transform

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
video_lock = threading.Lock()
# initialize the video stream and allow the camera sensor to
# warmup
video = VideoStream(src=0).start()
app = Flask(__name__)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


def capture_frames():
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, video_lock, video

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # apply face crop and histogram equalization
        frame = video.read()
        # frame = imutils.resize(frame, width=400)
        frame = crop_face_transform(frame)
        frame = histogram_equalization(frame)


        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (6, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # acquire the lock, set the output frame, and release the
        # lock
        with video_lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, video_lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with video_lock:
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
    # return the response generated along with the specific media type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start a thread that will perform motion detection
    t = threading.Thread(target=capture_frames)
    t.daemon = True
    t.start()

    app.run()


# release the video stream pointer
video.stop()
