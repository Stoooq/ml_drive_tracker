import json
from pathlib import Path

import cv2 as cv
import torchvision.models as tv_models

from src.data_manager import DataManager
from src.utils import (
    detect_and_draw_bboxes,
    get_model_classes_from_weights_meta,
    show_image_tensor,
)


def main():
    root_dir = Path("datasets/bdd100k")

    bb_model_weights = tv_models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    bb_model = tv_models.detection.fasterrcnn_resnet50_fpn(
        weights=bb_model_weights,
    ).eval()

    data_manager = DataManager(root_dir=root_dir, batch_size=32)
    train_loader, val_loader, test_loader = data_manager.build_dataloaders()

    num_classes, classes = get_model_classes_from_weights_meta(bb_model_weights)
    target_class_names = ["car", "traffic light"]
    bbox_colors = ["red", "blue"]
    object_indices = [classes.index(name) for name in target_class_names]

    confidence_threshold = 0.7

    train_dataset = train_loader.dataset
    first_image_name = train_dataset.labels[1]["name"]

    image_path = train_dataset.images_dir / first_image_name

    result_image_tensor = detect_and_draw_bboxes(
        model=bb_model,
        image_path=image_path,
        object_indices=object_indices,
        labels=target_class_names,
        bbox_colors=bbox_colors,
        threshold=confidence_threshold,
    )

    show_image_tensor(result_image_tensor)

    print("Hello from ml-drive-tracker!")


if __name__ == "__main__":
    main()
