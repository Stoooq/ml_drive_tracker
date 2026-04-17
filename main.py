from pathlib import Path

import cv2 as cv
import torchvision.models as tv_models
from torchvision import transforms

from data.data_manager import DataManager
from ml.model import (
    detect_and_draw_bboxes,
    get_model_classes_from_weights_meta,
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

    cap = cv.VideoCapture(
        "/Users/miloszglowacki/Desktop/code/python/ml_drive_tracker/istockphoto-2159760544-640_adpp_is.mp4"
    )

    last_boxes = []
    last_labels = []

    frame_count = 0
    process_every_n_frames = 5

    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            last_boxes = []

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            transform = transforms.Compose([transforms.ToTensor()])

            input_frame = transform(rgb_frame).unsqueeze(0)

            if not ret:
                print("Reading error.")
                break

            result_image_tensor, all_boxes_to_draw, all_labels_to_draw = (
                detect_and_draw_bboxes(
                    model=bb_model,
                    image_tensor=input_frame,
                    object_indices=object_indices,
                    labels=target_class_names,
                    bbox_colors=bbox_colors,
                    threshold=confidence_threshold,
                )
            )

            last_boxes.extend(all_boxes_to_draw)

            result_np = result_image_tensor.permute(1, 2, 0).cpu().numpy()

            result_bgr = cv.cvtColor(result_np, cv.COLOR_RGB2BGR)

            cv.imshow("Live preview", result_bgr)
        else:
            for box in last_boxes:
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                cv.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

            cv.imshow("Live preview", frame)

        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
