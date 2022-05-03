from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import io
import PIL.Image
import numpy as np
from statistics import mean

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[
            :-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# 0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}


    
def analyze_something(dataloader, cfg, writer):
    images = []
    boxes = {}
    dic = {}
    
    car = []
    bus = []
    background = []
    truck = []
    motorcycle =[]
    bicycle=[]
    scooter=[]
    person=[]
    rider = []
    
    car_ar = []
    bus_ar = []
    background_ar = []
    truck_ar = []
    motorcycle_ar =[]
    bicycle_ar=[]
    scooter_ar=[]
    person_ar=[]
    rider_ar = []
    
    car_area = []
    bus_area = []
    background_area = []
    truck_area = []
    motorcycle_area =[]
    bicycle_area=[]
    scooter_area=[]
    person_area=[]
    rider_area = []
    
    label_names = ['background', 'car','truck', 'bus', 'motorcycle','bicycle', 'scooter','person','rider']
    
    x=0
    aspect_ratios = []
    for batch in tqdm(dataloader):
#         Remove the two lines below and start analyzing :D
#         print("The keys in the batch are:", batch.keys())
#         exit()
#         print("Boxes:", batch['boxes'])
#         print("keys in batch:", batch.keys())
#         print("labels in batch:", batch['labels'])
        
        images.append(batch['image'])
        labels = batch['labels']
        boxes = batch["boxes"].tolist()
#         print("image size:", batch["image"].size())
        image_size_info= (batch["image"].size())
#         aspect_ratios.append(image_size_info[3]/image_size_info[2])
#         print("boxes utenfor", boxes)
        for box in boxes[0]:
            #  [x_min, y_min, x_max, y_max]
            h = box[2]-box[0]
            w=box[3]-box[1]
            aspect_ratios.append(w/h)

        for i, label in enumerate(labels.tolist()[0]):
            # add box with the label to the list of that kind of boxes
#             print("label:",label)
#             print("boxes innenfor", boxes[0][i])
            h = boxes[0][i][2]-boxes[0][i][0]
            w = boxes[0][i][3]-boxes[0][i][1]
    
            if label == 0:
                background.append(boxes[0][i])
                background_ar.append(w/h)
                background_area.append(w*h)
            if label == 1:
                car.append(boxes[0][i])
                car_ar.append(w/h)
                car_area.append(w*h)
            if label == 2:
                truck.append(boxes[0][i])
                truck_ar.append(w/h)
                truck_area.append(w*h)
            if label == 3:
                bus.append(boxes[0][i])
                bus_ar.append(w/h)
                bus_area.append(w*h)
            if label == 4:
                motorcycle.append(boxes[0][i])
                motorcycle_ar.append(w/h)
                motorcycle_area.append(w*h)
            if label == 5:
                bicycle.append(boxes[0][i])
                bicycle_ar.append(w/h)
                bicycle_area.append(w*h)
            if label == 6:
                scooter.append(boxes[0][i])
                scooter_ar.append(w/h)
                scooter_area.append(w*h)
            if label == 7:
                person.append(boxes[0][i])
                person_ar.append(w/h)
                person_area.append(w*h)
            if label == 8:
                rider.append(boxes[0][i])
                rider_ar.append(w/h)
                rider_area.append(w*h)
                
#         x+=1
#         if x>10:
#             break
    # create grid of images
#     print("background",background)
#     print("car",car)
#     print("rider",rider)
    
    print("background",len(background))
    print("car",len(car))
    print("truck", len(truck))
    print("rider", len(rider))
    print("bicycle",len(bicycle))
    print("motorcycle",len(motorcycle))
    print("person",len(person))
    print("bus",len(bus))
    print("scooter",len(scooter))
    
#     plt.switch_backend("agg")
    fig = plt.figure(0,figsize=(20,10))
    ax1 = plt.subplot()
    
    all_lists = [background,car,truck,bus,motorcycle,bicycle,scooter,person,rider]
    x_label_list = []
    for i, entry in enumerate(all_lists):
        plt.bar(i, len(entry))
        plt.text(i,len(entry),len(entry))
        x_label_list.append(i)
        
    ax1.set_xticks(x_label_list)    
    plt.xlabel("Labels")
    plt.ylabel("Number of instances")
    ax1.set_xticklabels(label_names)
    fig.savefig('dataset_exploration/bars-test.png')
    
    fig_hist = plt.figure(1)
    plt.hist(aspect_ratios, bins=100)
    plt.xlabel("Aspect ratios")
    plt.ylabel("Count")
    fig_hist.savefig('dataset_exploration/hist-test.png')
    
    fig_label_ar = plt.figure(2,figsize=(20,10))
    ars = [car_ar,bus_ar,background_ar,truck_ar,motorcycle_ar,bicycle_ar,scooter_ar,person_ar,rider_ar]
    means = []
    for ar in ars:
        if len(ar):
            means.append(mean(ar))
        else:
            means.append(0)
    
#     mean_result = [mean(car_ar),mean(bus_ar),(background_ar),mean(truck_ar),mean(motorcycle_ar),mean(bicycle_ar),mean(scooter_ar),mean(person_ar),mean(rider_ar)]
    plt.bar(label_names,means)
    plt.xlabel("Labels")
    plt.ylabel("Aspect ratio mean")
    fig_label_ar.savefig('dataset_exploration/ar-mean-test.png')
    
    fig_mean_area = plt.figure(3,figsize=(20,10))
    areas = [car_area,bus_area,background_area,truck_area,motorcycle_area,bicycle_area,scooter_area,person_area,rider_area]
    means_area = []
    for area in areas:
        if len(area):
            means_area.append(mean(area))
        else:
            means_area.append(0)
    
    plt.bar(label_names,means_area)

    
    plt.xlabel("Labels")
    plt.ylabel("Mean area")
    fig_mean_area.savefig('dataset_exploration/area-mean-test.png')
    
    

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
   
    
    writer = SummaryWriter("/home/karoliks/TDT4265_StarterCode/assignment4/SSD/tops/logger/testlog")
    analyze_something(dataloader, cfg, writer)
    writer.close()

if __name__ == '__main__':
    main()
