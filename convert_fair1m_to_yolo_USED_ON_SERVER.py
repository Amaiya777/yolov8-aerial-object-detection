import pandas as pd
import os
from tqdm import tqdm

# =========================
# PATHS
# =========================
BASE_PATH = "/workspace/Dataset"
LABELS_FILE = os.path.join(BASE_PATH, "labels.parquet")
OUTPUT_LABELS = os.path.join(BASE_PATH, "labels")

os.makedirs(os.path.join(OUTPUT_LABELS, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_LABELS, "val"), exist_ok=True)

# =========================
# LOAD DATA
# =========================
labels = pd.read_parquet(LABELS_FILE)

# =========================
# CATEGORY MAPPING (FINAL)
# =========================
def map_category(cat):
    cat = str(cat).lower()

    # Aircraft
    if any(x in cat for x in ["aircraft", "airplane", "plane", "a220"]):
        return 0

    # Ship
    elif any(x in cat for x in ["ship", "boat", "vessel"]):
        return 1

    # Vehicle (covers all variants)
    elif any(x in cat for x in [
        "vehicle", "car", "van", "bus", "truck",
        "dump truck", "cargo truck", "small car", "trailer"
    ]):
        return 2

    # Bridge
    elif "bridge" in cat:
        return 3

    # Harbor
    elif any(x in cat for x in ["harbor", "port"]):
        return 4

    elif "windmill" in cat:
        return 5

    elif "solar" in cat:
        return 6

    elif "tank" in cat:
        return 7

    elif "pool" in cat:
        return 8

    elif "tennis" in cat:
        return 9

    elif "basketball" in cat:
        return 10

    elif "football" in cat:
        return 11

    elif "baseball" in cat:
        return 12

    elif "parking" in cat:
        return 13

    elif "playground" in cat:
        return 14

    elif "crosswalk" in cat:
        return 15

    elif "street light" in cat:
        return 16

    elif "traffic light" in cat:
        return 17

    elif "traffic sign" in cat:
        return 18

    elif "billboard" in cat:
        return 19

    elif "fence" in cat:
        return 20

    elif "tree" in cat:
        return 21

    return None


# =========================
# CONVERT LABELS
# =========================
for _, row in tqdm(labels.iterrows(), total=len(labels)):

    split = str(row["Split"]).lower()

    image_name = os.path.basename(row["FilePath"])  
    image_id = os.path.splitext(image_name)[0]      
    

    cls = map_category(row["Category"])

    # Skip unknown categories safely
    if cls is None:
        continue

    xmin = row["x_min"]
    ymin = row["y_min"]
    xmax = row["x_max"]
    ymax = row["y_max"]

    img_w = row["ImageWidth"]
    img_h = row["ImageHeight"]

    # YOLO format conversion
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h

    label_line = f"{cls} {x_center} {y_center} {width} {height}\n"

    label_path = os.path.join(OUTPUT_LABELS, split, f"{image_id}.txt")

    with open(label_path, "a") as f:
        f.write(label_line)

print("✅ Conversion complete!")