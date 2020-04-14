import argparse
import cv2
import datetime
import requests

from image_processing import crop_face_transform
import utils


def send_frames(video, addr, size=(360, 240), fps=10, frames_to_send=-1, crop=False, print_interval=-1, log=None):
    crop_str = "True" if crop else "False"
    full_address = f"{addr}/cropped={crop_str}&width={size[0]}&height={size[1]}"

    send_forever = (frames_to_send == -1)

    frames_sent = 0

    fps_tracker = utils.FpsTracker(fps_limit=fps, average_over=10, print_interval=print_interval, log=log)
    fps_tracker.track()
    while frames_sent < frames_to_send or send_forever:
        success, frame = video.read()
        if not success:
            print("ERROR: Could not capture from web cam")
            break

        if crop:
            frame = crop_face_transform(frame)["img"]

        frame = cv2.resize(frame, dsize=size)

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
            requests.post(full_address, data=encodedImage.tostring(), headers=headers)
        except Exception as e:
            print(f"Error occurred while trying to send image:\n\t{e}\nExiting...")
            exit()

        frames_sent += 1
        fps_tracker.frame_sent()

    fps_tracker.close_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sends images to end point')
    parser.add_argument('-a', '--address', dest='address',
                        help="Web address to send the images captured from the camera to", required=True)
    parser.add_argument('-f', '--frames', dest='frames', default=-1, type=int,
                        help='Amount of frames from the camera to send to the endpoint (default: infinite)')
    parser.add_argument('--fps', default=-1, type=int, dest='fps',
                        help='FPS to capture images at (default: infinite)')
    parser.add_argument('-j', '--jetson', dest='jetson', action='store_true',
                        help='Flag specifying if you are running this from an nvidia jetson')
    parser.add_argument('-c', '--crop', dest='crop', action='store_true',
                        help='Flag specifying if you would like to perform face cropping before sending the image')
    parser.add_argument('-s', '--size', dest='size', nargs=2, type=int, default=[360, 240], metavar=('width', 'height'),
                        help='Specify what size the image sent should be (default: 360 x 240)')
    parser.add_argument('-p', '--print', dest='print', type=int, default=-1,
                        help='Specify what size the image sent should be')
    parser.add_argument('-l', '--log', dest='log', default=None,
                        help='File to log raw FPS data to')
    args = parser.parse_args()

    if args.jetson:
        video = cv2.VideoCapture(utils.jetson_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    else:
        video = cv2.VideoCapture(0)

    send_frames(video, args.address, size=tuple(args.size), fps=args.fps, frames_to_send=args.frames, crop=args.crop,
                print_interval=args.print, log=args.log)
    video.release()

