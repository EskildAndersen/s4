import h5py
import torch
from torch.utils.data import DataLoader, Dataset


class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_dict = {}

        # Load the data from HDF5 file
        with h5py.File(self.file_path, "r") as f:
            for key in f.keys():
                self.data_dict[key] = f[key][()]  # Load as numpy array

        self.keys = list(self.data_dict.keys())  # Store keys for indexing

    def __len__(self):
        # Return the total number of samples
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data = self.data_dict[key]  # Get the numpy array
        return torch.tensor(data)  # Convert to PyTorch tensor


if __name__ == "__main__":
    # Create the dataset
    file_path = "data/eeg/trainData.hdf5"
    dataset = HDF5Dataset(file_path)

    # Create the DataLoader
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example of iterating through the DataLoader
    for batch in data_loader:
        print(batch)  # Your batch of data
