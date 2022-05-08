import torchvision
from .task23_4 import ( # TODO endre hvis mer data augmentation
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


# anchors.aspect_ratios=[[1.2,3.5], [1.2, 3.5], [1.2, 3.5], [1.2, 3.5], [1.2,3.5], [1.2,3.5]]

# anchors.feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]]

# anchors.strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]]

anchors.min_sizes=[[16,4], [32, 8], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]] # test for  humans?
# anchors.min_sizes= [[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]] # test for  humans?

