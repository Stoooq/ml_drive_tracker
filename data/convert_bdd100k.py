import json
from pathlib import Path

CLASS_MAPPING = {"car": 0, "traffic light": 1}


def convert_split(
    json_path: Path,
    output_labels_dir: Path,
    target_class_names: list[str],
):
    with json_path.open() as f:
        json_data = json.load(f)

    output_labels_dir.mkdir(parents=True, exist_ok=True)

    for image in json_data:
        # print(image["name"], image["labels"])
        lines = []

        for obj in image["labels"]:
            if obj["category"] in target_class_names and "box2d" in obj:
                cx = (obj["box2d"]["x1"] + obj["box2d"]["x2"]) / 2 / 1280
                cy = (obj["box2d"]["y1"] + obj["box2d"]["y2"]) / 2 / 720
                w = (obj["box2d"]["x2"] - obj["box2d"]["x1"]) / 1280
                h = (obj["box2d"]["y2"] - obj["box2d"]["y1"]) / 720

                class_id = CLASS_MAPPING[obj["category"]]
                lines.append(f"{class_id} {cx} {cy} {w} {h}")

        label_file = output_labels_dir / Path(image["name"]).with_suffix(".txt")
        label_file.write_text("\n".join(lines))


if __name__ == "__main__":
    convert_split(
        Path(
            "datasets/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json",
        ),
        Path("datasets/bdd100k/bdd100k/bdd100k/labels/100k/train"),
        ["car", "traffic light"],
    )
    convert_split(
        Path(
            "datasets/bdd100k/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_val.json",
        ),
        Path("datasets/bdd100k/bdd100k/bdd100k/labels/100k/val"),
        ["car", "traffic light"],
    )
