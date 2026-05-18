import json
import multiprocessing
from pathlib import Path

import cv2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

from core.config import settings
from ml.model import ModelFormat, ObjectDetector


def evaluate_pytorch(model_path: Path):
    model = YOLO(str(model_path))

    metrics = model.val(data="coco.yaml")

    return {"map50": metrics.box.map50, "map": metrics.box.map}


def evaluate_model(
    model_format: ModelFormat,
    model_path: Path,
    images_dir: Path,
    annotations_path: Path,
):
    tmp_dir = Path("storage/output")

    coco_gt = COCO(str(annotations_path))
    images = coco_gt.loadImgs(coco_gt.getImgIds())

    cats = coco_gt.loadCats(coco_gt.getCatIds())
    name_to_cat_id = {cat["name"]: cat["id"] for cat in cats}

    batches = [images[i : i + 1000] for i in range(0, len(images), 1000)]

    for idx, batch in enumerate(batches):
        output_file = Path(tmp_dir / f"tmp_batch_{idx}.jsonl")

        p = multiprocessing.Process(
            target=_process_batch,
            args=(
                model_format,
                model_path,
                images_dir,
                name_to_cat_id,
                batch,
                output_file,
            ),
        )
        p.start()
        p.join()

    all_predictions = []
    for idx in range(len(batches)):
        with open(tmp_dir / f"tmp_batch_{idx}.jsonl") as f:
            for line in f:
                all_predictions.append(json.loads(line))

        Path(tmp_dir / f"tmp_batch_{idx}.jsonl").unlink()

    coco_dt = coco_gt.loadRes(all_predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    return {"map50": evaluator.stats[1], "map": evaluator.stats[0]}


def _process_batch(
    model_format: ModelFormat,
    model_path: Path,
    images_dir: Path,
    name_to_cat_id: dict[str, int],
    batch,
    output_path: Path,
):
    detector = ObjectDetector(
        model_format=model_format,
        model_path=model_path,
        bbox_colors=settings.bbox_colors,
        confidence_threshold=0.001,
        process_every_n_frames=settings.process_every_n_frames,
        tracking_enabled=False,
    )

    with open(output_path, "w") as f:
        for image in batch:
            img = cv2.imread(str(images_dir / image["file_name"]))
            detections = detector.detect(img)

            for detection in detections:
                w = detection.bbox[2] - detection.bbox[0]
                h = detection.bbox[3] - detection.bbox[1]

                f.write(
                    json.dumps(
                        {
                            "image_id": image["id"],
                            "category_id": name_to_cat_id[detection.class_name],
                            "bbox": [detection.bbox[0], detection.bbox[1], w, h],
                            "score": detection.confidence,
                        },
                    )
                    + "\n",
                )
