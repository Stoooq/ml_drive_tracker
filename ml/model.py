import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.models as tv_models
import torchvision.utils as vutils
from torchvision import transforms


class ObjectDetector:
    def __init__(
        self,
        target_classes: list[str],
        bbox_colors: list[str],
        confidence_threshold: float,
        process_every_n_frames: int,
        bbox_width: int = 3,
    ):
        self.target_classes = target_classes
        self.bbox_colors = bbox_colors
        self.confidence_threshold = confidence_threshold
        self.process_every_n_frames = process_every_n_frames
        self.bbox_width = bbox_width

        self.bb_model_weights = (
            tv_models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.bb_model = tv_models.detection.fasterrcnn_resnet50_fpn(
            weights=self.bb_model_weights,
        ).eval()

        self.num_classes, self.classes = self._get_model_classes_from_weights_meta()
        self.object_indices = [self.classes.index(name) for name in self.target_classes]

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def _get_model_classes_from_weights_meta(self):
        num_classes = None
        class_names = None

        if (
            self.bb_model_weights
            and hasattr(self.bb_model_weights, "meta")
            and "categories" in self.bb_model_weights.meta
        ):
            class_names = self.bb_model_weights.meta["categories"]
            num_classes = len(class_names)

            print(
                f"Model is configured for {num_classes} classes based on Weights Metadata. These classes are:\n",
            )
        else:
            print("'categories' metadata not found for this model.")

        return num_classes, class_names

    def _detect_and_draw_bboxes(
        self,
        image_tensor: torch.Tensor,
    ):
        result_image_tensor = (image_tensor.squeeze(0) * 255).byte()

        with torch.no_grad():
            prediction = self.bb_model(image_tensor)[0]

        all_boxes_to_draw = []
        all_labels_to_draw = []
        all_colors_to_draw = []

        for index, label, color in zip(
            self.object_indices, self.target_classes, self.bbox_colors, strict=True
        ):
            class_mask = (prediction["labels"] == index) & (
                prediction["scores"] > self.confidence_threshold
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
                width=self.bbox_width,
            )
        else:
            print(
                f"No objects from the list {self.target_classes} were found with a confidence score above {self.confidence_threshold}.\n",
            )

        return result_image_tensor, all_boxes_to_draw, all_labels_to_draw

    def detect_on_video(self, video_path: str, output_path: str):
        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

        last_boxes = []
        last_labels = []

        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1

            if not ret:
                print("Reading error.")
                break

            if frame_count % self.process_every_n_frames == 0:
                last_boxes = []
                last_labels = []

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                input_frame = self.to_tensor(rgb_frame).unsqueeze(0)

                result_image_tensor, all_boxes_to_draw, all_labels_to_draw = (
                    self._detect_and_draw_bboxes(image_tensor=input_frame)
                )

                last_boxes.extend(all_boxes_to_draw)
                last_labels.extend(all_labels_to_draw)

                result_np = result_image_tensor.permute(1, 2, 0).cpu().numpy()

                result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

                out.write(result_bgr)

            else:
                for box in last_boxes:
                    pt1 = (int(box[0]), int(box[1]))
                    pt2 = (int(box[2]), int(box[3]))
                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)

                out.write(frame)

        cap.release()
        out.release()

        return output_path
