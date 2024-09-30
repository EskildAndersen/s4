import h5py
import mne
import numpy as np
import torch
from scipy.signal import lfilter


class EEG_dataset_from_paths(torch.utils.data.Dataset):
    def __init__(
        self,
        bidsPaths,
        beforePts,
        afterPts,
        targetPts,
        channelIdxs=1,
        transform=None,
        hdf5File=None,
    ):
        if hdf5File is None:
            self.constructFromPaths(
                bidsPaths, beforePts, afterPts, targetPts, channelIdxs, transform
            )
        else:
            self.constructFromHDF5(hdf5File)

    def constructFromHDF5(self, hdf5File):
        # load all keys from hdf5 file into dataset:
        with h5py.File(hdf5File, "r") as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Group):
                    self.__dict__[key] = []
                    for subkey in f[key].keys():
                        self.__dict__[key].append(f[key][subkey][...])
                else:
                    if f[key].shape[0] == 1:
                        self.__dict__[key] = f[key][0]
                    else:
                        self.__dict__[key] = f[key][...]

        pass

    def constructFromPaths(
        self, bidsPaths, beforePts, afterPts, targetPts, channelIdxs=1, transform=None
    ):
        self.transform = transform
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.channelIdxs = channelIdxs
        self.nChannels = (
            len(channelIdxs) if isinstance(channelIdxs, (list, tuple, range)) else 1
        )
        self.file_paths = [str(fp) for fp in bidsPaths]

        self.step = 100  # how many samples should, at least, be between each data point

        maxFilesLoaded = self.determineMemoryCapacity()

        # preload:
        self.data_arrays = []
        nfilesToLoad = min(maxFilesLoaded, len(self.file_paths))
        fileIdxToLoad = np.random.choice(
            len(self.file_paths), nfilesToLoad, replace=False
        )

        for fileIdx in fileIdxToLoad:
            try:
                tempRaw = mne.io.read_raw_eeglab(
                    self.file_paths[fileIdx], preload=True, verbose=False
                )

                channelsToExclude = (
                    (1 - np.isin(range(0, tempRaw.info["nchan"]), self.channelIdxs))
                    .nonzero()[0]
                    .astype("int")
                )
                # import pdb; pdb.set_trace()
                channelsToExclude = np.asarray(tempRaw.ch_names)[channelsToExclude]
                tempRaw.drop_channels(channelsToExclude)

                if self.transform:
                    tempRaw = self.transform(tempRaw)

                self.data_arrays.append(tempRaw._data)
            except Exception as error:
                print(error)
                print(self.file_paths[fileIdx])

        self.getDataDraws()

    def getDataDraws(self):
        # find all allowed dataPoints:
        windowSize = self.beforePts + self.afterPts + self.targetPts
        allAllowedDataPoints = []

        def moving_average(data, windowSize):
            b = np.ones(windowSize) / windowSize
            a = 1
            return lfilter(b, a, data)

        for data in self.data_arrays:
            allowedDataPoints = np.zeros((data.shape[0], data.shape[1]), dtype=bool)
            for channelIdx in range(data.shape[0]):
                data_nan = np.isnan(data[channelIdx, :])
                filteredData = moving_average(data_nan[::-1], windowSize)[::-1]
                allowedDataPoints[channelIdx, :] = filteredData == 0

            allAllowedDataPoints.append(allowedDataPoints)

        # make three-column array with file, channel and index:
        self.dataDraws = []
        for fileIdx, allowedDataPoints in enumerate(allAllowedDataPoints):
            for channelIdx in range(0, allowedDataPoints.shape[0]):
                for i in range(
                    0, allowedDataPoints.shape[1] - windowSize, self.step
                ):  # start, stop, step
                    if allowedDataPoints[channelIdx, i]:
                        self.dataDraws.append([fileIdx, channelIdx, i])

        self.dataDraws = np.asarray(self.dataDraws)

    def updateDataSet(self, beforePts, afterPts, targetPts):
        """
        Syntax: updateDataSet(beforePts,afterPts,targetPts)
        """
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.getDataDraws()

    def determineMemoryCapacity(self):
        # determine how much space we can use for pre-loaded data:
        import psutil

        freeMemory = psutil.virtual_memory().available
        print("Detected free memory:", freeMemory / (1024**3), "GB")

        fileSizeMax = 10 * 3600 * 250  # 10 hours of data at 250Hz
        fileSizeMax = fileSizeMax * self.nChannels
        fileSizeMax *= 64 / 8  # size of a 10 hr night in bytes

        nFiles = int(freeMemory / fileSizeMax)
        print(
            "This will fit approximately %s files with %s  channels each"
            % (nFiles, self.nChannels)
        )
        print("")

        return nFiles

    def __len__(self):
        return self.dataDraws.shape[0]

    def __getitem__(self, idx):
        windowSize = self.beforePts + self.afterPts + self.targetPts

        fileIdx = self.dataDraws[idx, 0]
        channelIdx = self.dataDraws[idx, 1]
        sampleIdx = self.dataDraws[idx, 2]
        data_out = self.data_arrays[fileIdx][
            channelIdx, sampleIdx : sampleIdx + windowSize
        ]

        data_out = data_out * 1e6  # convert to microvolts

        # make sure there are no nan's in the data:
        assert not np.any(np.isnan(data_out))

        data_out = torch.tensor(data_out, dtype=torch.float32)
        data_out = data_out.reshape(-1, 1)  # nSamples x nChannels

        x12 = (data_out[0 : self.beforePts, :], data_out[-self.afterPts :, :])
        target = data_out[self.beforePts : (-self.afterPts), 0]
        # TODO: this needs to be updated if nChannels>1.
        # right now it assumes that we also want to predict channel 0

        return x12, target

    def saveToHDF5(self, hdf5File):
        """
        Save the dataset to a hdf5 file.
        Adds '.hdf5' to the end of the filename if it is not already there.

        """
        with h5py.File(hdf5File, "w") as f:
            for key, value in self.__dict__.items():
                if isinstance(value, list):
                    for idx, data in enumerate(value):
                        f.create_dataset(key + "/" + str(idx), data=data)
                elif isinstance(value, int):
                    f.create_dataset(key, data=value, shape=(1,))
                elif isinstance(value, range):
                    temp = np.asarray(value)
                    f.create_dataset(key, data=temp, shape=temp.shape)
                elif value is None:
                    f.create_dataset(key, data=value, shape=(0,))
                else:
                    f.create_dataset(key, data=value, shape=value.shape)


if __name__ == "__main__":
    dataset = EEG_dataset_from_paths(
        0, 0, 0, 0, hdf5File="C:/Program Files (x86)/s4/data/eeg/trainData.hdf5"
    )

    # beforePts, afterPts, targetPts
    dataset.updateDataSet(100, 100, 25)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for x, y in dataloader:
        print(x[0].shape)
        print(y.shape)
        break
