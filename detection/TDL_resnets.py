from typing import Optional, Callable, Type, Union, List, Any

import torch
import torchvision
from dropblock import DropBlock2D
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.models.resnet import conv3x3, conv1x1
from torchvision.models._api import WeightsEnum
from torchvision.models._utils import _ovewrite_named_param
from torchvision.utils import _log_api_usage_once

DROPBLOCK_PROB = 0.5
DROPOUT_PROB = 0.3
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class AvgPoolShortCut(nn.Module):
    """
    Strided average pooling as implemented for the DDU paper
    This module replaces the 1x1 convolution down-sampling in resnet layers
    """
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device,)
        x = torch.cat((x, pad), dim=1)
        return x


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        assert activation in ("relu", "leaky")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class BasicBlockSpectralNormalisation(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        assert activation in ("relu", "leaky")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_sn = nn.utils.parametrizations.spectral_norm(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.conv2_sn = nn.utils.parametrizations.spectral_norm(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1_sn(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2_sn(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, BasicBlockSpectralNormalisation]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        half_sn: Optional[bool] = False,
        activation: Optional[str] = "relu",
        avg_pool: Optional[bool] = False,
        dropblock_location: int = 2
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        assert activation in ("relu", "leaky")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.avg_pool = avg_pool
        self.dropblock_location = dropblock_location
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        # Build with top half of the layers with spectral normalisation
        self.half_sn = half_sn
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], activation=activation)
        # Layer 2
        if self.half_sn and self.dropblock_location == 1:
            self.layer2 = self._make_layer(BasicBlockSpectralNormalisation, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0], activation=activation)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], activation=activation)
        # Layer 3
        if self.half_sn and self.dropblock_location <= 2:
            self.layer3 = self._make_layer(BasicBlockSpectralNormalisation, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1], activation=activation)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1],
                                           activation=activation)
        # Layer 4
        if self.half_sn and self.dropblock_location <= 3:
            self.layer4 = self._make_layer(BasicBlockSpectralNormalisation, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2], activation=activation)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], activation=activation)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockSpectralNormalisation) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, BasicBlockSpectralNormalisation]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        activation: str = "relu",
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_pool:
                downsample = nn.Sequential(AvgPoolShortCut(stride, block.expansion * planes, self.inplanes))
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, activation,
            )
        )
        self.inplanes = planes * block.expansion

        # if stride != 1 or self.inplanes != planes * block.expansion:
        #     if self.avg_pool:
        #         downsample = nn.Sequential(AvgPoolShortCut(stride, block.expansion * planes, self.inplanes))
        #     else:
        #         downsample = nn.Sequential(
        #             conv1x1(self.inplanes, planes * block.expansion, stride),
        #             norm_layer(planes * block.expansion),
        #         )
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    # stride=stride,
                    # downsample=downsample,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    activation=activation
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, BasicBlockSpectralNormalisation]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model


def create_model(spectral_normalization: bool = False,
                 half_sn: bool = False,
                 activation: str = "relu",
                 avg_pool: bool = False,
                 num_classes: int = 1000,
                 dropblock_location: int = 2,
                 original_architecture: bool = False) -> ResNet:
    if spectral_normalization:
        model = _resnet(
            block=BasicBlockSpectralNormalisation,
            layers=[2, 2, 2, 2],
            weights=None,
            progress=True,
            activation=activation,
            avg_pool=avg_pool,
            num_classes=num_classes,
            dropblock_location=dropblock_location
        )
    else:
        model = _resnet(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            weights=None,
            progress=True,
            half_sn=half_sn,
            activation=activation,
            avg_pool=avg_pool,
            num_classes=num_classes,
            dropblock_location=dropblock_location
        )
    # else:
    #     model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    if original_architecture:
        if spectral_normalization:
            model.conv1 = nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            )
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        if spectral_normalization:
            model.conv1 = nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            )
        else:
            model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        model.maxpool = nn.Identity()
    return model


class LitResnet(LightningModule):
    def __init__(self,
                 lr=0.05,
                 num_classes: int = 1000,
                 spectral_normalization: bool = False,
                 fifth_conv_layer: bool = False,
                 extra_fc_layer: bool = False,
                 dropout: bool = False,
                 dropblock: bool = False,
                 half_sn: bool = False,
                 activation: str = "relu",
                 dropblock_prob: Optional[float] = 0.5,
                 dropblock_size: Optional[int] = 3,
                 dropblock_location: Optional[int] = 2,
                 dropout_prob: Optional[float] = 0.5,
                 avg_pool: Optional[bool] = False,
                 loss_type: str = "nll",
                 optimizer_type: str = "sgd",
                 original_architecture: bool = False):
        super().__init__()
        assert dropblock_location in (1, 2, 3)
        assert loss_type in ("nll", "ce")
        assert optimizer_type in ("sgd", "adam")
        self.save_hyperparameters()
        self.model = create_model(spectral_normalization=spectral_normalization,
                                  half_sn=half_sn,
                                  activation=activation,
                                  avg_pool=avg_pool,
                                  num_classes=num_classes,
                                  dropblock_location=dropblock_location,
                                  original_architecture=original_architecture)
        self.fifth_conv_layer = fifth_conv_layer
        self.extra_fc_layer = extra_fc_layer
        self.dropout = dropout
        self.dropblock = dropblock
        self.dropblock_location = dropblock_location
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        if self.dropblock:
            self.model.dropblock_layer = DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size)
        if self.fifth_conv_layer:
            self.model.layer5 = nn.Sequential(
                conv3x3(512, 512, groups=1, dilation=1),
                nn.ReLU(),
                DropBlock2D(dropblock_prob, block_size=dropblock_size)
            )
        if self.extra_fc_layer:
            self.model.fc_dropout = nn.Linear(512, 512)
        if self.dropout:
            self.model.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # out = self.model(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.activation(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        if self.dropblock and self.dropblock_location == 1:
            x = self.model.dropblock_layer(x)
        x = self.model.layer2(x)
        if self.dropblock and self.dropblock_location == 2:
            x = self.model.dropblock_layer(x)
        x = self.model.layer3(x)
        if self.dropblock and self.dropblock_location == 3:
            x = self.model.dropblock_layer(x)
        x = self.model.layer4(x)
        if self.fifth_conv_layer:
            x = self.model.layer5(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        if self.extra_fc_layer:
            x = self.model.fc_dropout(x)
        if self.dropout:
            x = self.model.dropout(x)
        x = self.model.fc(x)
        if self.loss_type == "nll":
            return F.log_softmax(x, dim=1)
        else:
            return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if self.loss_type == "nll":
            loss = F.nll_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        if self.loss_type == "nll":
            loss = F.nll_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        if self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
                weight_decay=5e-4,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=5e-4
            )
        steps_per_epoch = 74000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
