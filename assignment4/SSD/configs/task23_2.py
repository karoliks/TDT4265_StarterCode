import torchvision
from .task23_1 import ( 
    train,
    optimizer,
    schedulers,
#     loss_objective,
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
from ssd.modeling.focal_loss import FocalLoss

# 0.01 for background class, 1 for the rest
loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=[0.01,*[1 for i in range(model.num_classes-1)]], gamma=2, num_classes=model.num_classes) 