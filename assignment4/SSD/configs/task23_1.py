import torchvision
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
#     backbone,
    data_train,
    data_val,
    val_cpu_transform,
    label_map,
    anchors,
    min_sizes,
    train_cpu_transform
)
import torchvision
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from tops.config import LazyCall as L
from ssd.data.mnist import MNISTDetectionDataset
from ssd import utils
from ssd.data.transforms import Normalize, ToTensor, GroundTruthBoxesToAnchors
from .utils import get_dataset_dir, get_output_dir
from ssd.modeling.backbones import FPN


backbone = L(FPN)(
    output_channels=[256, 256, 256, 256, 256, 256, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)
