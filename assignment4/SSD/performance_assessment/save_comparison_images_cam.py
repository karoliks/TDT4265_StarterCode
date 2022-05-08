import cv2
import os
import tops
import click
import numpy as np
from tops.config import instantiate
from tops.config import LazyCall as L
from tops.checkpointer import load_checkpoint
from vizer.draw import draw_boxes
from ssd import utils
from tqdm import tqdm
from ssd.data.transforms import ToTensor
from pytorch_grad_cam import AblationCAM, EigenCAM
# from pytorch_grad_cam.utils import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import torch
from torchvision.models import resnet34
import inspect
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
from collections import OrderedDict
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform


    
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model

    def forward(self,x):
        to_return = self.model(x.cuda())[0]
        return torch.Tensor(to_return[0].cpu())


def fpn_reshape_transform(x):

    return x[0]



def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        x1 = round(x1)
        y1 = round(y1)
        x2 = round(x2)
        y2 = round(y2)
        
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    return eigencam_image_renormalized




def get_config(config_path):
    cfg = utils.load_config(config_path)
    cfg.train.batch_size = 1
    cfg.data_train.dataloader.shuffle = False
    cfg.data_val.dataloader.shuffle = False
    return cfg


def get_trained_model(cfg):
    model = tops.to_cuda(instantiate(cfg.model))
    model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    model.load_state_dict(ckpt["model"])
    return model



def get_dataloader(cfg, dataset_to_visualize):
    # We use just to_tensor to get rid of all data augmentation, etc...
    to_tensor_transform = [
        L(ToTensor)()
    ]
    if dataset_to_visualize == "train":
        cfg.data_train.dataset.transform.transforms = to_tensor_transform
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataset.transform.transforms = to_tensor_transform
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def convert_boxes_coords_to_pixel_coords(boxes, width, height):
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes.cpu().numpy()


def convert_image_to_hwc_byte(image):
    first_image_in_batch = image[0]  # This is the only image in batch
    image_pixel_values = (first_image_in_batch * 255).byte()
    image_h_w_c_format = image_pixel_values.permute(1, 2, 0)
    return image_h_w_c_format.cpu().numpy()


def visualize_annotations_on_image(image, batch, label_map):
    boxes = convert_boxes_coords_to_pixel_coords(batch["boxes"][0], batch["width"], batch["height"])
    labels = batch["labels"][0].cpu().numpy().tolist()

    image_with_boxes = draw_boxes(image, boxes, labels, class_name_map=label_map)
    return image_with_boxes


def visualize_model_predictions_on_image(image, img_transform, batch, model, label_map, score_threshold):

    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]

    boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
    categories = categories.cpu().numpy().tolist()

    image_with_predicted_boxes = draw_boxes(image, boxes, categories, scores, class_name_map=label_map)
    return image_with_predicted_boxes


def visualize_cam_on_image(image, img_transform, batch, model, label_map, score_threshold):
    
    image_float_np = np.float32(image) / 255
    pred_image = tops.to_cuda(batch["image"])
    transformed_image = img_transform({"image": pred_image})["image"]
    
    wrapped_model = Wrapper(model)
    target_layers = [model.feature_extractor]

    cam = EigenCAM(wrapped_model, 
            target_layers, 
            use_cuda=torch.cuda.is_available(),
            reshape_transform=fpn_reshape_transform
    )

    grayscale_cam = cam(transformed_image)   
    grayscale_cam = grayscale_cam[0, :]
    
    
    cam_image = show_cam_on_image(image_float_np,grayscale_cam, use_rgb=True)
    
    

    
    
    
#     pred_image = tops.to_cuda(batch["image"])


    boxes, categories, scores = model(transformed_image, score_threshold=score_threshold)[0]
    boxes = convert_boxes_coords_to_pixel_coords(boxes.detach().cpu(), batch["width"], batch["height"])
    categories = categories.cpu().numpy().tolist()
    
    cam_image = renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam)

    image_with_predicted_boxes = draw_boxes(cam_image, boxes, categories, scores, class_name_map=label_map)
    
    return image_with_predicted_boxes





def create_filepath(save_folder, image_id):
    filename = "image_" + str(image_id) + ".png"
    return os.path.join(save_folder, filename)


def create_comparison_image(batch, model, img_transform, label_map, score_threshold):
    image = convert_image_to_hwc_byte(batch["image"])
    image_with_annotations = visualize_annotations_on_image(image, batch, label_map)
    image_with_model_predictions = visualize_model_predictions_on_image(
        image, img_transform, batch, model, label_map, score_threshold)
    image_with_cam = visualize_cam_on_image(
        image, img_transform, batch, model, label_map, score_threshold)
    concatinated_image = np.concatenate([
        image,
        image_with_annotations,
        image_with_model_predictions,
        image_with_cam
    ], axis=0)
    return concatinated_image


def create_and_save_comparison_images(dataloader, model, cfg, save_folder, score_threshold, num_images):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Saving images to", save_folder)

    num_images_to_save = min(len(dataloader), num_images)
    dataloader = iter(dataloader)

    img_transform = instantiate(cfg.data_val.gpu_transform)
    for i in tqdm(range(num_images_to_save)):
        batch = next(dataloader)
        comparison_image = create_comparison_image(batch, model, img_transform, cfg.label_map, score_threshold)
        filepath = create_filepath(save_folder, i)
        cv2.imwrite(filepath, comparison_image[:, :, ::-1])


def get_save_folder_name(cfg, dataset_to_visualize):
    return os.path.join(
        "performance_assessment",
        cfg.run_name,
        dataset_to_visualize
    )


@click.command()
@click.argument("config_path")
@click.option("--train", default=False, is_flag=True, help="Use the train dataset instead of val")
@click.option("-n", "--num_images", default=500, type=int, help="The max number of images to save")
@click.option("-c", "--conf_threshold", default=0.3, type=float, help="The confidence threshold for predictions")
def main(config_path, train, num_images, conf_threshold):
    cfg = get_config(config_path)
    model = get_trained_model(cfg)

    if train:
        dataset_to_visualize = "train"
    else:
        dataset_to_visualize = "val"

    dataloader = get_dataloader(cfg, dataset_to_visualize)
    save_folder = get_save_folder_name(cfg, dataset_to_visualize)

    create_and_save_comparison_images(dataloader, model, cfg, save_folder, conf_threshold, num_images)


if __name__ == '__main__':
    main()
