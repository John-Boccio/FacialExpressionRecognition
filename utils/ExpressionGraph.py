import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2

import utils


class ExpressionGraph(object):
    def __init__(self, exp_class=utils.FerExpression):
        self.fig = plt.figure()
        plt.ylabel('Probability')
        plt.xlabel('Frames')
        plt.suptitle('Expressions Probability Distribtion')
        self.ax = self.fig.add_subplot(111)
        self.exp_class = exp_class
        self.num_exp = len(self.exp_class)
        self.exp_names = [e.name for e in self.exp_class]
        self.lines = []
        for i in range(self.num_exp):
            p, = self.ax.plot([], [], label=self.exp_names[i])
            self.lines.append(p)
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))
        self.frame_counter = 0

    def update(self, exp_pdist):
        self.frame_counter += 1
        for i in range(len(self.lines)):
            self.lines[i].set_xdata(np.append(self.lines[i].get_xdata(), self.frame_counter))
            self.lines[i].set_ydata(np.append(self.lines[i].get_ydata(), exp_pdist[i]))
        self.ax.relim()
        self.ax.autoscale_view()

    def get_img(self, ):
        buf = io.BytesIO()
        self.fig.savefig(buf, format="jpg", bbox_inches='tight')
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img

