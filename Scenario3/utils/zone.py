#zone.py
import cv2
from utils import config
import threading

# 구역 표시
def annotate_zones(displayed_frame):
    cv2.rectangle(displayed_frame, config.warning_zone_start, config.warning_zone_end, (0, 0, 0), 3)
    cv2.putText(displayed_frame, "warning_zone", (config.warning_zone_end[0] - 60, config.warning_zone_end[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(displayed_frame, config.danger_zone_start, config.danger_zone_end, (0, 0, 255), 3)
    cv2.putText(displayed_frame, "danger_zone", (config.danger_zone_end[0] - 100, config.danger_zone_end[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    return displayed_frame

#구역 내 손 검출
def is_inside_danger_zone(box, zone_start, zone_end):
    x_min, y_min, x_max, y_max = box
    return (x_min < zone_end[0] and x_max > zone_start[0] and
            y_min < zone_end[1] and y_max > zone_start[1])
