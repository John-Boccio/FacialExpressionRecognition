import argparse
import cv2
import datetime
import requests
import time

from imutils.video import VideoStream


def send_frames(video, fps=10, frames_to_send=-1):
    mspf = (1/fps)*1000
    time_ms = lambda: int(round(time.time()*1000))
    send_forever = (frames_to_send == -1)
    frames_sent = 0
    global_st_time = time.time()
    while frames_sent < frames_to_send or send_forever:
        st_time = time_ms()
        frame = video.read()

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        (success, encodedImage) = cv2.imencode(".jpg", frame)
        if not success:
            continue

        headers = {'content-type': 'image/jpeg'}
        requests.post("http://localhost:5000/video_feed", data=encodedImage.tostring(), headers=headers)
        frames_sent += 1

        if frames_sent % fps == 0:
            end_time = time.time()
            print(f"Estimated FPS\t{frames_sent/(end_time-global_st_time)}")

        end_time = time_ms()
        elapsed_ms = end_time - st_time
        if elapsed_ms < mspf:
            time.sleep((mspf-elapsed_ms)/1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends images to end point')
    parser.add_argument('-a', '--address', dest='address', metavar='ADDR',
                        help="Web address to send the images captured from the camera to", required=True)
    parser.add_argument('-f', '--frames', dest='frames', default=-1, type=int, metavar='F',
                        help='Amount of frames from the camera to send to the endpoint (default: infinite)')
    parser.add_argument('--fps', default=15, type=int, dest='fps', metavar='FPS',
                        help='FPS to capture images at (default: 15)')
    args = parser.parse_args()
    video = VideoStream(src=0).start()
    send_frames(video, fps=args.fps, frames_to_send=args.frames)

