import os
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.stats as st

import pyedflib
import sys

# Read EDF file and print some usefull data
def read_edf_file(file):
    f = pyedflib.EdfReader(file)
    if True:
        print("signallabels: %s" % f.getSignalLabels())
        print("singnals in file: %s" % f.signals_in_file)
        print("datarecord duration: %f seconds" % f.getFileDuration())
        print("samplefrequency: %f" % f.getSampleFrequency(1))
        # print("samplefrequency: %f" % f.getSampleFrequency(27))
        # f._close()
    return f

read_edf_file('testdata/2p11.edf')