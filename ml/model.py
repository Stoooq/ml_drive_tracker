import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils


def collate_fn(batch):
    return tuple(zip(*batch, strict=True))


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
