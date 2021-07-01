import typing

import torch


# Pre-activation ResNet code is ported from https://github.com/locuslab/robust_overfitting/blob/master/preactresnet.py


# noinspection PyAbstractClass
class PreActBlock(torch.nn.Module):
    EXPANSION = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super(PreActBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False
        )

        if stride != 1 or in_channels != self.EXPANSION * out_channels:
            self.shortcut = torch.nn.Conv2d(
                in_channels, self.EXPANSION * out_channels, kernel_size=(1, 1), stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out = out + shortcut
        return out


# noinspection PyAbstractClass
class PreActBottleneck(torch.nn.Module):
    EXPANSION = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int
    ):
        super(PreActBottleneck, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(out_channels)
        self.conv3 = torch.nn.Conv2d(
            out_channels, self.EXPANSION * out_channels, kernel_size=(1, 1), bias=False
        )

        if stride != 1 or in_channels != self.EXPANSION * out_channels:
            self.shortcut = torch.nn.Conv2d(
                in_channels, self.EXPANSION * out_channels, kernel_size=(1, 1), stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch.relu(self.bn2(out)))
        out = self.conv3(torch.relu(self.bn3(out)))
        out = out + shortcut
        return out


# noinspection PyAbstractClass
class PreActResNet(torch.nn.Module):

    def __init__(
            self,
            block_cls: typing.Type[torch.nn.Module],
            num_blocks: typing.Tuple[int, int, int, int],
            num_classes: int,
            image_channels: int
    ):
        super(PreActResNet, self).__init__()

        base_channels = 64
        self.conv1 = torch.nn.Conv2d(
            image_channels, base_channels, kernel_size=(3, 3), padding=1, bias=False
        )
        in_channels = base_channels
        self.layer1, in_channels = self._make_layer(
            in_channels, base_channels, num_blocks[0], stride=1, block_cls=block_cls
        )
        self.layer2, in_channels = self._make_layer(
            in_channels, 2 * base_channels, num_blocks[1], stride=2, block_cls=block_cls
        )
        self.layer3, in_channels = self._make_layer(
            in_channels, 4 * base_channels, num_blocks[2], stride=2, block_cls=block_cls
        )
        self.layer4, in_channels = self._make_layer(
            in_channels, 8 * base_channels, num_blocks[3], stride=2, block_cls=block_cls
        )
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.linear = torch.nn.Linear(in_channels, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.relu(self.bn(out))
        out = torch.nn.functional.avg_pool2d(out, kernel_size=(4, 4))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def _make_layer(
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int,
            block_cls: typing.Type[torch.nn.Module]
    ) -> typing.Tuple[torch.nn.Module, int]:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block_cls(in_channels=in_channels, out_channels=out_channels, stride=stride))
            in_channels = block_cls.EXPANSION * out_channels
        return torch.nn.Sequential(*layers), in_channels


class PreActResNet18(PreActResNet):
    def __init__(self, num_classes: int, image_channels: int = 3):
        super(PreActResNet18, self).__init__(
            block_cls=PreActBlock,
            num_blocks=(2, 2, 2, 2),
            num_classes=num_classes,
            image_channels=image_channels
        )


class PreActResNet34(PreActResNet):
    def __init__(self, num_classes: int, image_channels: int = 3):
        super(PreActResNet34, self).__init__(
            block_cls=PreActBlock,
            num_blocks=(3, 4, 6, 3),
            num_classes=num_classes,
            image_channels=image_channels
        )


class PreActResNet50(PreActResNet):
    def __init__(self, num_classes: int, image_channels: int = 3):
        super(PreActResNet50, self).__init__(
            block_cls=PreActBottleneck,
            num_blocks=(3, 4, 6, 3),
            num_classes=num_classes,
            image_channels=image_channels
        )


class PreActResNet101(PreActResNet):
    def __init__(self, num_classes: int, image_channels: int = 3):
        super(PreActResNet101, self).__init__(
            block_cls=PreActBottleneck,
            num_blocks=(3, 4, 23, 3),
            num_classes=num_classes,
            image_channels=image_channels
        )


class PreActResNet152(PreActResNet):
    def __init__(self, num_classes: int, image_channels: int = 3):
        super(PreActResNet152, self).__init__(
            block_cls=PreActBottleneck,
            num_blocks=(3, 8, 36, 3),
            num_classes=num_classes,
            image_channels=image_channels
        )


# noinspection PyAbstractClass
class MLP(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_hidden: typing.Iterable[int],
            activation: typing.Optional[torch.nn.Module]
    ):
        super(MLP, self).__init__()

        layer_list = []

        # Add hidden layers
        for layer_width in num_hidden:
            layer_list.append(torch.nn.Linear(in_features, layer_width, bias=True))
            in_features = layer_width
            if activation is not None:
                layer_list.append(activation)

        # Add output layer
        layer_list.append(torch.nn.Linear(in_features, out_features, bias=True))

        self.layers = torch.nn.Sequential(*layer_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
