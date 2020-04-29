import time


class FpsTracker(object):
    def __init__(self, name="FPS Tracker", fps_limit=-1, average_over=10, print_interval=-1, log=None, flush_interval=1):
        """
        FpsTracker allows you to track the amount of frames you have sent and limit your FPS. The FPS is calculated by
        using a moving average over n samples.

        :param name: name used when printing
        :param fps_limit: specify the maximum FPS you'd like to process, if <= 0 then it will run as fast as possible
        :param average_over: number of samples to use for moving average
        :param print_interval: frame_sent will print the current fps statistics if this amount of time in ms has passed
            since the last time it printed. Use -1 if you would not like to print.
        :param log: file to log raw FPS data to, will not perform logging if 'None'.
        """
        self.name = name
        self.fps_limit = fps_limit if fps_limit > 0 else -1
        self.average_over = average_over
        self.print_interval = print_interval
        self.last_print_time = 0
        self.last_frame_sent_time = 0
        self.fps = 0
        self.moving_avg = []
        self.frames_sent = 0
        self.frames_per_ms = self.fps_limit / 1000
        self.ms_per_frame = 1 / self.frames_per_ms
        self.log_file = log
        self.log = None if log is None else open(log, 'w')
        self.flush_interval = flush_interval

        self.current_time_ms = lambda: int(round(time.time()*1000))

    def frame_sent(self):
        """
        Call after sending a frame. Each frame is allotted 'self.ms_per_frame' amount of time to send. If the frame has
        been sent in less than that amount of time, it will sleep the remainder.
        """
        self.frames_sent += 1

        current_time = self.current_time_ms()
        time_elapsed = current_time - self.last_frame_sent_time

        if time_elapsed < self.ms_per_frame:
            # Sleep the remainder of the given time
            seconds_remaining = (self.ms_per_frame - time_elapsed) / 1000
            time.sleep(seconds_remaining)
        tmp_time = self.last_frame_sent_time
        self.last_frame_sent_time = self.current_time_ms()

        instantaneous_fps = (1 / (self.current_time_ms() - tmp_time)) * 1000
        self.moving_avg.append(instantaneous_fps)
        if self.frames_sent < self.average_over:
            self.fps = (self.fps * (self.frames_sent - 1) + instantaneous_fps) / self.frames_sent
        else:
            self.fps = sum(self.moving_avg) / self.average_over
            self.moving_avg.pop(0)

        if self.log is not None:
            self.log.write(f"{instantaneous_fps:4.4f}, {self.fps:4.4f}\r\n")
            if (self.frames_sent % self.flush_interval) == 0:
                self.log.flush()

        if self.print_interval >= 0:
            time_elapsed = current_time - self.last_print_time
            if time_elapsed > self.print_interval:
                print(f"{self.name} - Frames sent: {self.frames_sent:6d}\tFPS: {self.fps:4.4f}")
                self.last_print_time = self.current_time_ms()

    def track(self):
        self.last_frame_sent_time = self.current_time_ms()
        self.last_print_time = self.current_time_ms()

    def reset_fps(self):
        self.fps = 0
        self.frames_sent = 0
        self.moving_avg = []

    def close_log(self):
        if self.log is not None:
            self.log.close()

