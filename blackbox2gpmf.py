import subprocess
import re
import csv

import numpy as np
import pandas as pd
from scipy import signal
from shutil import copyfile
from tkinter import filedialog, messagebox

import matplotlib
from matplotlib.widgets import Button, Slider
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
fig.canvas.set_window_title('blackbox2gpmf by jaromeyer')

gp_gyro, gp_offset, bbl_gyro, bbl_frame = [[]] * 4
bbl_plot, gp_file, gp_offsets = [None] * 3


def load_gp(event):
    global gp_gyro, gp_offsets, gp_file, bbl_plot
    gp_file = filedialog.askopenfilename(
        title="Select video", filetypes=(("GoPro MP4 Files", "*.mp4"),))

    # open gp file
    try:
        gp = open(gp_file, "rb")
    except FileNotFoundError:
        print("file not found")
    else:
        # read gopro file & find offsets
        gp_array = gp.read()
        gp.close()
        gp_batch_offsets = [match.start() for match in re.finditer(
            b'\x47\x59\x52\x4f', gp_array)]

        gp_gyro = []
        gp_offsets = []

        # loop over samples & save gyro readings as gp_gyro[sample][axis]
        for batch_offset in gp_batch_offsets:
            batch_nof_sample = int.from_bytes(
                gp_array[batch_offset + 6:batch_offset + 8], "big")
            for sample in range(batch_nof_sample):
                sample_offset = batch_offset + sample * 6 + 8
                x = int.from_bytes(
                    gp_array[sample_offset:sample_offset + 2], byteorder='big', signed=True)
                y = int.from_bytes(
                    gp_array[sample_offset + 2:sample_offset + 4], byteorder='big', signed=True)
                z = int.from_bytes(
                    gp_array[sample_offset + 4:sample_offset + 6], byteorder='big', signed=True)
                gp_gyro.append([x, y, z])
                gp_offsets.append(sample_offset)

        # clear axis and draw both plots
        ax.cla()
        bbl_gyro = []
        bbl_plot = None
        ax.plot([sample[0] for sample in gp_gyro], "g")
        plt.draw()


def load_bbl(event):
    global bbl_gyro

    if len(gp_gyro) == 0:
        messagebox.showinfo(
            title="Done", message="Please load GoPro file first")
        return

    # open bbl file
    bbl_file = filedialog.askopenfilename(
        title="Select bbl", filetypes=(("Blackbox logs", "*.csv"),))
    try:
        bbl = open(bbl_file)
    except FileNotFoundError:
        print("file not found")
    else:
        # find header
        gyro_index = None
        csv_reader = csv.reader(bbl)
        for i, row in enumerate(csv_reader):
            if(row[0] == "loopIteration"):
                gyro_index = row.index('gyroADC[0]')
                break
        bbl_df = pd.read_csv(bbl, header=None)
        bbl.close()

        camera_model = subprocess.getoutput(
            "exiftool -s -s -s -FirmwareVersion " + gp_file)[:3]

        if camera_model == "HD5":
            samplerate = 402
        else:
            samplerate = 197

        bbl_duration = (bbl_df[1].iloc[-1] - bbl_df[1].iloc[0]) / 1000 / 1000
        bbl_nof_samples = int(bbl_duration*samplerate)

        # downsample & save as bbl_gyro[sample][axis]
        bbl_x = np.clip(signal.resample(
            [-33*i for i in bbl_df[gyro_index].tolist()], bbl_nof_samples), a_min=-32768, a_max=32767)
        bbl_y = np.clip(signal.resample(
            [-33*i for i in bbl_df[gyro_index+1].tolist()], bbl_nof_samples), a_min=-32768, a_max=32767)
        bbl_z = np.clip(signal.resample(
            [33*i for i in bbl_df[gyro_index+2].tolist()], bbl_nof_samples), a_min=-32768, a_max=32767)
        bbl_gyro = np.column_stack((bbl_x, bbl_y, bbl_z))

        update_frame(0)
        draw_bbl_plot()


def patch(event):
    # check if both gyros have been loaded
    if len(gp_gyro) == 0 or len(bbl_gyro) == 0:
        messagebox.showinfo(title="Error", message="Please load files first")
        return

    # generate and open output_file
    output_file = gp_file[:-4] + "_bbl.MP4"
    copyfile(gp_file, output_file)
    gp = open(output_file, "rb+")

    # loop over gp_offsets and write bbl_gyro
    for i, sample in enumerate(gp_offsets):
        x = int(bbl_frame[i][0])
        y = int(bbl_frame[i][1])
        z = int(bbl_frame[i][2])

        gp.seek(sample)
        gp.write(x.to_bytes(2, byteorder='big', signed=True))
        gp.seek(sample + 2)
        gp.write(y.to_bytes(2, byteorder='big', signed=True))
        gp.seek(sample + 4)
        gp.write(z.to_bytes(2, byteorder='big', signed=True))
    gp.close()

    # change metadata to fake a hero6
    subprocess.run(['exiftool', '-FirmwareVersion=HD6.01.01.60.00',
                    '-Model="HERO6 Black"', '-overwrite_original', output_file])
    messagebox.showinfo(
        title="Done", message="Successfully patched %s" % output_file)


def draw_bbl_plot():
    global bbl_plot

    ydata = [sample[0] for sample in bbl_frame]

    # draw plot
    if bbl_plot == None:
        bbl_plot, = ax.plot(ydata, "r")
    else:
        bbl_plot.set_ydata(ydata)
    plt.draw()


def update_frame(offset):
    global bbl_gyro, bbl_frame

    offset = int(offset)

    # pad zeros right if the requested frame overflows bbl_gyro
    if offset + len(gp_gyro) > len(bbl_gyro):
        bbl_gyro = np.concatenate((bbl_gyro, [[0, 0, 0]] * (
            offset + len(gp_gyro) - len(bbl_gyro))))

    # pad zeros left if offset is negative
    if offset < 0:
        bbl_frame = np.concatenate(
            ([[0, 0, 0]] * -offset, bbl_gyro))[:len(gp_gyro)]
    else:
        bbl_frame = bbl_gyro[offset:offset + len(gp_gyro)]
    draw_bbl_plot()


# GUI
load_gp_axis = plt.axes([0.5, 0.05, 0.125, 0.05])
load_gp_button = Button(load_gp_axis, 'Load GP')
load_gp_button.on_clicked(load_gp)

load_bbl_axis = plt.axes([0.65, 0.05, 0.125, 0.05])
load_bbl_button = Button(load_bbl_axis, 'Load BBL')
load_bbl_button.on_clicked(load_bbl)

patch_axis = plt.axes([0.8, 0.05, 0.1, 0.05])
patch_btn = Button(patch_axis, 'Patch')
patch_btn.on_clicked(patch)

offset_axis = plt.axes([0.125, 0.125, 0.775, 0.05])
offset_slider = Slider(offset_axis, 'Offset', -1000, 2000, valstep=1)
offset_slider.on_changed(update_frame)

plt.show()
