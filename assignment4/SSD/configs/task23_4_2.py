import torchvision
from .task23_3 import ( # TODO endre hvis mer data augmentation
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    label_map,
    anchors,
    min_sizes,
    train_cpu_transform
)
from tops.config import LazyCall as L
from ssd.modeling.retinanet import RetinaNet
from ssd.modeling import AnchorBoxes

# From piazza:
# If you want to use the same classification and regression networks at every layer, then there must be the same amount of bounding boxes at each anchor location at every feature map, which can be achieved by making "aspect_ratios" for the anchor boxes the same at every feature map.
anchors.aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]]


model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8+1,  # Add 1 for background
    anchor_prob_initialization=True # A way to turn off and on the improved wheught initialization
)

from .utils import get_dataset_dir
data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")
