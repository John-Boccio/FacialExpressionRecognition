import argparse
import cv2
import datetime
import requests
import time

from image_processing import crop_face_transform
import utils


def send_frames(video, addr, fps=10, frames_to_send=-1, crop=False):
    # Minimum number of milliseconds we should take to process 1 frame to achieve <= 'fps'
    mspf = (1/fps)*1000
    time_ms = lambda: int(round(time.time()*1000))
    send_forever = (frames_to_send == -1)
    frames_sent = 0
    global_st_time = time.time()
    crop_str = "True" if crop else "False"
    while frames_sent < frames_to_send or send_forever:
        st_time = time_ms()
        success, frame = video.read()
        if not success:
            print("ERROR: Could not capture from web cam")
            break

        if crop:
            frame = crop_face_transform(frame)["img"]

        frame = cv2.resize(frame, dsize=(360, 240))

        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        (success, encodedImage) = cv2.imencode(".jpg", frame)
        if not success:
            continue

        headers = {'content-type': 'image/jpeg'}
        try:
            requests.post(addr + "/cropped=" + crop_str, data=encodedImage.tostring(), headers=headers)
        except Exception as e:
            print(f"Error occurred while trying to send image:\n\t{e}\nExiting...")
            exit()
        frames_sent += 1

        if frames_sent % fps == 0:
            end_time = time.time()
            print(f"Estimated FPS\t{frames_sent/(end_time-global_st_time)}")

        end_time = time_ms()
        elapsed_ms = end_time - st_time
        # If we finished the frame faster than our minimum time per frame, sleep until we reach that time
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
    parser.add_argument('-j', '--jetson', dest='jetson', action='store_true',
                        help='Flag specifying if you are running this from an nvidia jetson')
    parser.add_argument('-c', '--crop', dest='crop', action='store_true',
                        help='Flag specifying if you would like to perform face cropping before sending the image')
    args = parser.parse_args()

    if args.jetson:
        video = cv2.VideoCapture(utils.jetson_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        video = cv2.VideoCapture(0)

    send_frames(video, addr=args.address, fps=args.fps, frames_to_send=args.frames, crop=args.crop)
    video.release()

