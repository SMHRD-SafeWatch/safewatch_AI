from ultralytics import YOLO

def get_model():
    model = YOLO('models/best.pt')
    model.conf = 0.75
    model.iou = 0.45
    return model