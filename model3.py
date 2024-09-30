# more elaborate version with multi steps of decoding

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_utils import EEG_dataset_from_paths
from models.s4.s4d import S4D

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split(".")[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split(".")[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class EEGpredictor(L.LightningModule):
    def __init__(
        self,
        beforePts,
        afterPts,
        targetPts,
        nChannels,
        d_input=1,
        d_output=1,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        lr=0.01,
        weight_decay=0.01,
        prenorm=False,
    ):
        super(EEGpredictor, self).__init__()
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.nChannels = nChannels

        self.prenorm = prenorm
        self.lr = lr
        self.weight_decay = weight_decay

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        # TODO We get 1 x for beforePts and 1 x afterPts needs to be handled
        x = self.encoder(x[0])  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log(
            "train_MSE", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        self.log(
            "val_MSE", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# unit tests:
if __name__ == "__main__":

    def test_forward_pass():
        # test the forward pass:
        nBatch = 64
        beforePts = 100
        afterPts = 100
        targetPts = 25

        model = EEGpredictor(beforePts, afterPts, targetPts, 1)

        x = (torch.randn(nBatch, 100, 1), torch.randn(nBatch, 100, 1))

        y_hat = model(x)
        assert y_hat.shape == (nBatch, targetPts)

    def test_training_step():
        # test the training step:
        model = EEGpredictor(100, 100, 25, 1)
        x = (torch.randn(64, 100, 1), torch.randn(64, 100, 1))
        y = torch.randn(64, 25)
        loss = model.training_step((x, y), 0)
        assert loss > 0

        loss.backward()

    def training_loop_lightning():
        # test the training loop:
        model = EEGpredictor(100, 100, 25, 1)
        trainer = L.Trainer(max_epochs=1, accelerator="cpu", devices=1)

        class fakeDataSet(torch.utils.data.Dataset):
            def __init__(self):
                pass

            def __len__(self):
                return 64

            def __getitem__(self, idx):
                return (torch.randn(100, 1), torch.randn(100, 1)), torch.randn(25)

        dataloader = torch.utils.data.DataLoader(fakeDataSet(), batch_size=64)
        trainer.fit(model, dataloader)

    # torch.autograd.set_detect_anomaly(True)
    training_loop_lightning()

    dataset = EEG_dataset_from_paths(
        0, 0, 0, 0, hdf5File="C:/Program Files (x86)/s4/data/eeg/trainData.hdf5"
    )

    beforePts = 250
    afterPts = 250
    targetPts = 100

    # beforePts, afterPts, targetPts
    dataset.updateDataSet(beforePts, afterPts, targetPts)

    model = EEGpredictor(beforePts, afterPts, targetPts, 1)

    # validationSize=1e2
    trainingSize = 1e3
    # nEpochs=100

    # #use SubsetRandomSampler to split the data into training and validation sets of limited size:
    trainLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=torch.utils.data.SubsetRandomSampler(range(int(trainingSize))),
    )

    trainer = L.Trainer(max_epochs=5)

    trainer.fit(model, trainLoader)
