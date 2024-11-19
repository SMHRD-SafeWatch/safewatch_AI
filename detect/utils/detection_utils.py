# utils/detection_utils.py
def check_overlap(region1, region2):
    """두 영역이 겹치는지 확인"""
    x1, y1, x2, y2 = region1
    x3, y3, x4, y4 = region2
    return not ((x1 >= x4) or (x2 <= x3) or (y1 >= y4) or (y2 <= y3))

def check_vertical_stack(detections, threshold=6, y_gap=150, x_gap=100):
    """박스가 수직으로 일정 수 이상 쌓여 있는지 확인"""
    box_centers = []
    for box in detections['box']:
        x1, y1, x2, y2 = box['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_centers.append((center_x, center_y))

    if not box_centers:
        return False

    # 박스 중심 좌표를 x, y로 정렬
    box_centers.sort(key=lambda b: (b[0], b[1]))  # x 기준 정렬 후 y 기준 정렬

    # 수직 스택 확인
    stack_count = 1  # 초기 스택 카운트
    for i in range(1, len(box_centers)):
        x_diff = abs(box_centers[i][0] - box_centers[i - 1][0])  # x 간격
        y_diff = abs(box_centers[i][1] - box_centers[i - 1][1])  # y 간격

        if x_diff <= x_gap and y_diff <= y_gap:  # x, y 간격 모두 충족해야 스택으로 간주
            stack_count += 1
            if stack_count >= threshold:
                return True
        else:
            stack_count = 1  # 스택 초기화

    return False

def check_irregular_stack(detections, x_gap=100, min_boxes=3):
    """
    박스가 불규칙적으로 쌓여 있는지 확인
    - x_gap: 허용되는 수평 간격
    - min_boxes: 불규칙 스택으로 간주할 최소 박스 수
    """
    box_centers = []
    for box in detections['box']:
        x1, y1, x2, y2 = box['bbox']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_centers.append((center_x, center_y))

    if len(box_centers) < min_boxes:
        return False  # 박스가 충분하지 않으면 불규칙 스택 아님

    # y 기준으로 정렬
    box_centers.sort(key=lambda b: b[1])

    # 박스 간 x 간격 확인
    for i in range(1, len(box_centers)):
        x_diff = abs(box_centers[i][0] - box_centers[i - 1][0])
        if x_diff > x_gap:
            return True  # 수평 간격이 허용 범위를 초과하면 불규칙 스택

    return False
