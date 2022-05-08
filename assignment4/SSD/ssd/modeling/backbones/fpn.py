import torch
import torchvision
import torchvision.models as models
from typing import Tuple, List
import pprint


class FPN(torch.nn.Module):

    def __init__(self,
                 output_channels: List[int],
                 image_channels: int,
                 output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        resnet = models.resnet50(pretrained=True)
        
        # Want to remove the last to layars to remove classification.
        self.resnet_parts = torch.nn.ModuleList(list(resnet.children())[:-2])
        self.fpn = torchvision.ops.FeaturePyramidNetwork([256, 512, 1024, 2048, 256, 256], 256)

        # What we replace the last two layers with
        part5 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=2048,
                out_channels=output_channels[4],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=output_channels[4],
                out_channels=output_channels[5],
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.ReLU(),

        )
        part6 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=output_channels[5],
                out_channels=output_channels[6],
                kernel_size=2,
                stride=1,
                padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=output_channels[6],
                out_channels=output_channels[7],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            torch.nn.ReLU(),

        )
        self.resnet_parts.append(part5)
        self.resnet_parts.append(part6)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_keys =['feat0','feat1', 'feat2', 'feat3', 'feat4', 'feat5']

        for i,part in enumerate(self.resnet_parts):
            x = part(x)
            if i > 3: # Resnet model starts with conv1, bn1, relu and maxpool. We want to skip these
                out_features.append(x)
                
        # Make dict for fpn
        out_features_dict = dict(zip(out_keys, out_features))
        
        # Use it on fpn
        out_features_dict = self.fpn(out_features_dict)
        out_features = out_features_dict.values()
        

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

