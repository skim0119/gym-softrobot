"""
Created on Sep. 23, 2021
@author: Heng-Sheng (Hanson) Chang
"""

import os, shutil
import matplotlib.pyplot as plt

class FrameBase(object):
    def __init__(self, figure_name, folder_name=None, fig_dict=None):
        self.figure_name = figure_name
        self.folder_name = folder_name
        self.frame_count = 0
        self.fig_dict = fig_dict
        self.check_folder()

    def check_folder(self,):
        if not (self.folder_name is None):
            if os.path.exists(self.folder_name):
                print('Clean up files in: {}/'.format(self.folder_name))
                shutil.rmtree(self.folder_name)
            print('Create the directory: {}/'.format(self.folder_name))
            os.mkdir(self.folder_name)

    def reset(self):
        self.fig = plt.figure(**self.fig_dict)

    def save(self, show=False, frame_count=None):
        if self.folder_name is None:
            self.fig.savefig(self.figure_name)
        else:
            frame_count = (
                self.frame_count if frame_count is None else frame_count
            )
            self.fig.savefig(
                self.folder_name + "/" +
                self.figure_name.format(frame_count)
            )
            self.frame_count += 1
        if show:
            plt.show()
        else:
            plt.close(self.fig)

    def movie(self, frame_rate, movie_name):
        print("Create movie:", movie_name+".mov")
        cmd = "ffmpeg -r {}".format(frame_rate)
        figure_name = self.figure_name.replace("{:", "%")
        figure_name = figure_name.replace("}", "")
        cmd += " -i " + self.folder_name + "/" + figure_name
        cmd += " -b:v 90M -c:v libx264 -pix_fmt yuv420p -f mov"
        cmd += " -y " + movie_name + ".mov"
        os.system(cmd)
