# import comet_ml
from ultralytics import YOLO

# comet_ml.login(project_name="comet-test-gajahdet")

# Load a model
model = YOLO("yolov8n.pt")

# Customize validation settings
metrics = model.val(
    data="coco.yaml", 
    imgsz=640, 
    batch=16, 
    conf=0.25, 
    iou=0.6, 
    device="0,1",
    save_json=True,
    project="comet-obsdet"
    )

# print(metrics.box.map)  # mAP50-95
# print(metrics.box.map50)  # mAP50
# print(metrics.box.map75)  # mAP75
# print(metrics.box.maps)  # list of mAP50-95 for each category