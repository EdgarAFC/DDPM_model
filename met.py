# import OS module
import os

import csv

import numpy as np
import h5py
import os

from scipy.signal import hilbert


class PlaneWaveData:
    """ A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]

    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError
        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.idata.shape == self.qdata.shape
        assert self.idata.ndim == self.qdata.ndim == 3
        nangles, nchans, nsamps = self.idata.shape
        assert self.angles.ndim == 1 and self.angles.size == nangles
        assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        assert self.time_zero.ndim == 1 and self.time_zero.size == nangles
        # print("Dataset successfully loaded")

class LoadData_nair2020(PlaneWaveData):
    def __init__(self, h5_dir, simu_name):
        # raw_dir = 'D:\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw'
        # raw_dir = 'D:\\Itamar\\datasets\\fieldII\\simulation\\nair2020\\raw12500_0.5attenuation'
        simu_number = int(simu_name[4:])
        lim_inf = 1000*((simu_number-1)//1000) + 1
        lim_sup = lim_inf + 999
        h5_name = 'simus_%.5d-%.5d.h5' % (lim_inf, lim_sup)
        h5filename = os.path.join(h5_dir, h5_name)
        # print(h5filename)
        with h5py.File(h5filename, "r") as g:
        # g = h5py.File(filename, "r")
            f = g[simu_name]
            self.idata = np.expand_dims(np.array(f["signal"], dtype="float32"), 0)
            self.qdata = np.imag(hilbert(self.idata, axis=-1))
            self.angles = np.array([0])
            self.fc = np.array(f['fc']).item()
            self.fs = np.array(f['fs']).item()
            self.c = np.array(f['c']).item()
            self.time_zero = np.array([np.array(f['time_zero']).item()])
            self.fdemod = 0
            xs = np.squeeze(np.array(f['ele_pos']))
            self.grid_xlims = [xs[0], xs[-1]]
            self.grid_zlims = [30*1e-3, 80*1e-3]
            self.ele_pos = np.array([xs, np.zeros_like(xs), np.zeros_like(xs)]).T
            self.pos_lat = np.array(f['lat_pos']).item()
            self.pos_ax = np.array(f['ax_pos']).item()
            self.radius = np.array(f['r']).item()
        super().validate()

    

def main():

    # Get the list of all files and directories
    path = "/mnt/nfs/efernandez/datasets/dataRF/RF_test/"
    dir_list = os.listdir(path)
    print("Files and directories in '", path, "' :")
    # prints all files
    print(dir_list)

    for i in range(0, len(dir_list)):
        dir_list[i] = dir_list[i][:-4]

    # field names
    fields = [' ', 'id', 'r', 'cx', 'cz', 'c']

    rows = []
    n_sample = 0

    for simu in dir_list:
        sub_row = []
        P = LoadData_nair2020(h5_dir='/nfs/privileged/isalazar/datasets/simulatedCystDataset/raw_0.0Att/',
                            simu_name=simu)
        sub_row.append(n_sample)
        sub_row.append(int(simu[4:]))
        sub_row.append(P.radius)
        sub_row.append(P.pos_lat)
        sub_row.append(P.pos_ax)
        sub_row.append(P.c)

        rows.append(sub_row)

    # name of csv file
    filename = "/mnt/nfs/efernandez/datasets/test_sim_parameters.csv"
 
    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
 
        # writing the fields
        csvwriter.writerow(fields)
 
        # writing the data rows
        csvwriter.writerows(rows)

if __name__ == '__main__':
    main()
    



