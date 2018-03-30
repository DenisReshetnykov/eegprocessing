import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as st

import pyedflib
import sys
# import pyeeg

# Read EDF file and print some usefull data
def read_edf_file(file):
    f = pyedflib.EdfReader(file)
    print("signallabels: %s" % f.getSignalLabels())
    print("singnals in file: %s" % f.signals_in_file)
    print("datarecord duration: %f seconds" % f.getFileDuration())
    print("samplefrequency: %f" % f.getSampleFrequency(1))
    signal = f.readSignal(1)
    print(signal)
    s = open('eeg_s.txt', 'w')
    for index in signal[0:200]:
        s.write(str(index)+'\n')
    s.close()
    return f

read_edf_file('test.edf')

# Clear Axis from unnecessary labels and spines
def clear_axis():
    frame = plt.gca()
    # frame.axes.get_yaxis().set_ticks([])  # clear y axis
    frame.spines["top"].set_visible(False)
    frame.spines["bottom"].set_visible(False)
    frame.spines["right"].set_visible(False)
    frame.spines["left"].set_visible(False)
    frame.axes.tick_params(axis='x',labelsize=8)

# Add grid to plot
def add_grid(f,channel):
    # minor_ticks = np.arange(-500, 500, 20)
    # plt.gca().axes.set_yticks(minor_ticks, minor=True)
    plt.gca().axes.grid(color='blue', linestyle='--', axis='x', which='both')
    plt.gca().axes.grid(color='blue', linestyle='--', axis='y', which='major')
    plt.gca().axes.grid(color='blue', linestyle=':', axis='y', which='minor', alpha=0.4)
    plt.gca().yaxis.grid(True, which='both')
    plt.gca().xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: x / f.getSampleFrequency(channel)))


# Plot signals on different subplots
def plot_on_subplots(channels, file):
    f = read_edf_file(file)
    i = 0
    # stat_scratch(channels, f)
    for channel in channels:
        i+=1
        mpl.rcParams['axes.titlesize'] = 'small'
        signal = f.readSignal(channel-1)
        ax = plt.subplot(len(channels), 2, i*2-1)
        plt.title(f.getLabel(channel-1)+'         ', position=(0, 0.3), ha='right')
        clear_axis()
        plt.plot(signal, linewidth=0.2, color='black')
        # window_plot(signal, ax, f.getSampleFrequency(channel), 10000)
        add_grid(f, channel)

        # Compute and plot spectral analyzis for signal (hardcoded for that time, needto be parametrized)
        ax_spectr = plt.subplot(len(channels), 2, i * 2)
        ax_spectr.set_xscale('log')
        Power, PowerRatio, PowerFreq = spectral_analyzis(signal, 250, [0.5, 4, 8, 12, 35], 301000, 301500, 10)
        band_color=['red','blue','green','yellow']
        for color_iter in range(4):
            plt.fill_between(PowerFreq[int(len(PowerFreq)/4)*color_iter:int(len(PowerFreq)/4)*(color_iter+1)], PowerRatio[int(len(PowerRatio)/4)*color_iter:int(len(PowerRatio)/4)*(color_iter+1)], alpha=0.3, linewidth=0.4, color=band_color[color_iter])


        #correlation analysis
        # ax_correlation


# Plot signal window as patch (some hardcode in
def window_plot(signal, ax, Freq, size):
    '''
    :param signal: signal for maximum and minmum boundaries
    :param ax: plot for wich window must be added
    :param Freq: frequencies to compute Power in different rhytm band
    :param size: x-length of ploted window
    :return:
    '''
    window = patches.Rectangle((0, min(signal)), size, (max(signal) - min(signal)), alpha=0.1, fc='yellow')
    ax.add_patch(window)
    handler = EventHandler(window)
    Power, Power_Ratio = bin_power(signal, [0.5, 4, 8, 12, 30], Freq)
    text = '  alpha= '+str(Power_Ratio[2])+'\n'+'  beta= '+str(Power_Ratio[3])+'\n'+'  delta= '+str(Power_Ratio[0])+'\n'+'  theta= '+str(Power_Ratio[1])
    plt.annotate(text, xy=(0, 0), xytext=(len(signal), 0))


def spectral_analyzis(Signal, SignalFreq, Band, EpochStart=0, EpochStop=None, DFreq=1):
    '''
    :param Signal: list, 1-D real signal
    :param EpochStart: integer,
    :param EpochStop: integer,
    :param SignalFreq: integer, Signal physical frequency
    :param Band: list, real frequencies (in Hz) of bins  Each element of Band is a physical frequency and shall not exceed the Nyquist frequency, i.e., half of sampling frequency.
    :param DFreq: integer, number of equal segments in each band
    :return: Power: list, 2-D power in each Band divided on equal DFreq segments
    :return: PowerRatio: spectral power in each segment normalized by total power in ALL frequency bins.
    :return: PowerFreq: Frequencies in wich Power computed
    '''
    SignalSection = Signal[EpochStart:EpochStop]
    fftSignal = abs(np.fft.fft(SignalSection))
    Power = np.zeros((len(Band) - 1)*DFreq)
    PowerFreq = np.zeros((len(Band) - 1)*DFreq)

    for BandIndex in range(0, len(Band) - 1):
        Freq = float(Band[BandIndex])
        NextFreq = float(Band[BandIndex + 1])
        for FreqDiff in range(1, DFreq+1):
            FreqD=Freq+(NextFreq-Freq)*((FreqDiff-1)/DFreq)
            NextFreqD=Freq+(NextFreq-Freq)*(FreqDiff/DFreq)
            Power[BandIndex*DFreq+FreqDiff-1] = sum(fftSignal[int(np.floor(FreqD / SignalFreq * len(SignalSection))) : int(np.floor(NextFreqD / SignalFreq * len(SignalSection)))])
            PowerFreq[BandIndex*DFreq+FreqDiff-1] = FreqD
    PowerRatio = Power / sum(Power)
    return Power, PowerRatio, PowerFreq



# Plot signals on one plot with y-offset 100
def plot_on_one(channels,file):
    f = read_edf_file(file)
    clear_axis()
    i = 0
    for channel in channels:
        i+=1
        signal = f.readSignal(channel-1)
        plt.plot(signal-100*i, linewidth=0.1, color='black')
        plt.text(0, (-100)*i, f.getLabel(channel-1)+'     ', ha='right', color='black', size = 8)
        add_grid(f, channel)

# Find signal power in delta(0.5-4 Hz), theta(4-8 Hz), alpha(8-12 Hz), beta(12-30 Hz) frequency ranges from start to stop
def signal_power(channels,file, start=0, stop=None):
    f = read_edf_file(file)
    for channel in channels:
        signal = f.readSignal(channel)
        Power, Power_Ratio = bin_power(signal[start:stop], [0.5, 4, 8, 12, 30], f.getSampleFrequency(channel))
        # print(Power)
        # print(Power_Ratio)
    return Power_Ratio


class EventHandler(object):
    def __init__(self,window):
        fig.canvas.mpl_connect('button_press_event', self.onpress)
        fig.canvas.mpl_connect('button_release_event', self.onrelease)
        fig.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.x0, self.y0 = window.xy
        self.pressevent = None
        print("init")

    def onpress(self, event):
        print("onpress1")
        if event.inaxes != ax:
            return print('onpress')

        if not window.contains(event)[0]:
            return print('no onpress')

        self.pressevent = event

    def onrelease(self, event):
        self.pressevent = None
        self.x0, self.y0 = window.xy

    def onmove(self, event):
        if self.pressevent is None or event.inaxes != self.pressevent.inaxes:
            return

        dx = event.xdata - self.pressevent.xdata
        dy = event.ydata - self.pressevent.ydata
        window.xy = self.x0 + dx, self.y0 + dy
        line.set_clip_path(window)
        fig.canvas.draw()





# ReadEDFfile("test.edf")

# fig = plt.figure(figsize=(20,10))
# channels = np.arange(n, m + 1, 1): #for signal range [n;m]



# channels = [1,3,7,27]
channels = [1,2]
plot_on_subplots(channels,file = "test.edf")

# plt.show()








# plot_on_one([1],"test.edf")
# signal_power([1,2],"test.edf")










# Для научных публикаций цитатник
# 1. mathplotlib
# @Article{Hunter:2007,
#   Author    = {Hunter, J. D.},
#   Title     = {Matplotlib: A 2D graphics environment},
#   Journal   = {Computing In Science \& Engineering},
#   Volume    = {9},
#   Number    = {3},
#   Pages     = {90--95},
#   abstract  = {Matplotlib is a 2D graphics package used for Python
#   for application development, interactive scripting, and
#   publication-quality image generation across user
#   interfaces and operating systems.},
#   publisher = {IEEE COMPUTER SOC},
#   doi = {10.1109/MCSE.2007.55},
#   year      = 2007
# }
#
#
#
# 2. PyEEG
# Forrest S. Bao, Xin Liu and Christina Zhang, "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction," Computational Intelligence and Neuroscience, March, 2011