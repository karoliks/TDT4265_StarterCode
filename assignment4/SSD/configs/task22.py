# Import everything from the old dataset and only change the dataset folder.
import torchvision
from .tdt4265 import (
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
    min_sizes
)
from .utils import get_dataset_dir
from ssd.data import TDT4265Dataset ## todo trenger vi denne?
from tops.config import LazyCall as L # TODO trenger vi denne?
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomHorizontalFlip)

# TODO inspo https://piazza.com/class/kyipdksfp9q1dn?cid=389

# be careful when looking at aguemtneation from torchvisoin. a lot are for images and not lidar data. TODO fjerne kommentar
train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
        # you cannot repalce this with torchiviosn resize. need to wrap it(?) TODO fjerne kommentar
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    
        # all augmentetaion have to be done before this last line TODO fjerne kommentar
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

# currentlu only doign normalizaiton here. yuo donth have to calculate mean and std unless yo wany to calcualte it for train dataset.
# dont augment validation dataset TODO fjerne kommetnarer
# transforms that are easily prallellizable should be run on gpu. usually image-centred transorms. TODO fjerne komemtnar
# do not run transofrms that can renduce communcation(?)
# guess on randomhorizontalcropping. is porbably faster on cpu.
gpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
])

# Do not change, validation!
# val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
#     L(ToTensor)(),
#     L(Resize)(imshape="${train.imshape}"),
# ])

data_train.dataset = L(TDT4265Dataset)(
    img_folder=get_dataset_dir("tdt4265_2022"),
    transform="${train_cpu_transform}",
    annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json"))

# Do not change, validation!
# data_val.dataset = L(TDT4265Dataset)(
#     img_folder=get_dataset_dir("tdt4265_2022"),
#     transform="${val_cpu_transform}",
#     annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json"))
# data_val.gpu_transform = gpu_transform

data_train.gpu_transform = gpu_transform