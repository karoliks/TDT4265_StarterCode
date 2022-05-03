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

