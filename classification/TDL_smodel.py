from torch import nn
from dropblock import DropBlock2D


class FashionCNN(nn.Module):

    def __init__(self,
                 dropblock_1: bool,
                 dropblock_2: bool,
                 dropout: bool,
                 dropblock_1_prob: float = 0.3,
                 dropblock_2_prob: float = 0.25,
                 dropblock_1_size: int = 3,
                 dropblock_2_size: int = 3,
                 dropout_prob: float = 0.5,
                 leaky_relu: bool = False,
                 spectral_normalization: bool = False,
                 average_pooling: bool = False
                 ):
        super(FashionCNN, self).__init__()
        self.dropout = dropout
        self.dropblock_1 = dropblock_1
        self.dropblock_2 = dropblock_2
        self.dropblock_1_prob = dropblock_1_prob
        self.dropblock_2_prob = dropblock_2_prob
        self.layer1 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
            ) if spectral_normalization else nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) if average_pooling else nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            ) if spectral_normalization else nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) if average_pooling else nn.MaxPool2d(kernel_size=2, stride=2)
        )
        if self.dropblock_1:
            self.dropblock1 = DropBlock2D(drop_prob=self.dropblock_1_prob, block_size=dropblock_1_size)
        self.layer3 = nn.Sequential(
            nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            ) if spectral_normalization else nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.LeakyReLU() if leaky_relu else nn.ReLU(),
        )
        if self.dropblock_2:
            # Dropblock for catching the OoD
            self.dropblock2 = DropBlock2D(drop_prob=self.dropblock_2_prob, block_size=dropblock_2_size)
        self.fc1 = nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.relu1 = nn.ReLU()
        if self.dropout:
            self.drop = nn.Dropout2d(dropout_prob)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        if self.dropblock_1:
            out = self.dropblock1(out)
        out = self.layer3(out)
        if self.dropblock_2:
            out = self.dropblock2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        if self.dropout:
            out = self.drop(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
