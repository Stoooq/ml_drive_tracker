import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


def collate_fn(batch):
    return tuple(zip(*batch, strict=True))


def calculate_mean_std(dataset):
    preprocess = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ],
    )

    total_pixels = 0
    sum_pixels = torch.zeros(3)

    mean_loader = tqdm(dataset, desc="Pass 1/2: Computing Mean")

    for img, _ in mean_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        sum_pixels += pixels.sum(dim=1)
        total_pixels += pixels.size(1)

    mean = sum_pixels / total_pixels

    sum_squared_diff = torch.zeros(3)

    std_loader = tqdm(dataset, desc="Pass 2/2: Computing Std")

    for img, _ in std_loader:
        img_tensor = preprocess(img)
        pixels = img_tensor.view(3, -1)
        diff = pixels - mean.unsqueeze(1)
        sum_squared_diff += (diff**2).sum(dim=1)

    std = torch.sqrt(sum_squared_diff / total_pixels)

    return mean, std


def get_transforms(dataset: Dataset | None, mean: float | None, std: float | None):
    if not mean and not std:
        mean, std = calculate_mean_std(dataset)

    augmentations_transforms = [
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.RandomVerticalFlip(p=0.5),
    ]

    main_transforms = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    augmentation_transform = transforms.Compose(
        main_transforms + augmentations_transforms,
    )
    transform = transforms.Compose(main_transforms)

    return augmentation_transform, transform


def get_model_classes_from_weights_meta(weights_obj=None):
    num_classes = None
    class_names = None

    if (
        weights_obj
        and hasattr(weights_obj, "meta")
        and "categories" in weights_obj.meta
    ):
        class_names = weights_obj.meta["categories"]
        num_classes = len(class_names)

        print(
            f"Model is configured for {num_classes} classes based on Weights Metadata. These classes are:\n",
        )

        return num_classes, class_names
    else:
        print("'categories' metadata not found for this model.")
        return num_classes, class_names


def show_image_tensor(img_tensor: torch.Tensor, title: str = "Wykryte obiekty"):
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(12, 8))
    plt.imshow(img_np)
    plt.title(title)
    plt.axis("off")
    plt.show()


def detect_and_draw_bboxes(
    model,
    image_tensor,
    object_indices,
    labels,
    bbox_colors,
    threshold,
    bbox_width=3,
):
    result_image_tensor = (image_tensor.squeeze(0) * 255).byte()

    with torch.no_grad():
        prediction = model(image_tensor)[0]

    all_boxes_to_draw = []
    all_labels_to_draw = []
    all_colors_to_draw = []

    for index, label, color in zip(object_indices, labels, bbox_colors, strict=True):
        class_mask = (prediction["labels"] == index) & (
            prediction["scores"] > threshold
        )

        boxes_for_this_class = prediction["boxes"][class_mask]

        if boxes_for_this_class.nelement() > 0:
            all_boxes_to_draw.extend(boxes_for_this_class.tolist())
            all_labels_to_draw.extend([label] * len(boxes_for_this_class))
            all_colors_to_draw.extend([color] * len(boxes_for_this_class))

    if all_boxes_to_draw:
        result_image_tensor = vutils.draw_bounding_boxes(
            result_image_tensor,
            torch.tensor(all_boxes_to_draw),
            labels=all_labels_to_draw,
            colors=all_colors_to_draw,
            width=bbox_width,
        )
    else:
        print(
            f"No objects from the list {labels} were found with a confidence score above {threshold}.\n",
        )

    return result_image_tensor, all_boxes_to_draw, all_labels_to_draw
