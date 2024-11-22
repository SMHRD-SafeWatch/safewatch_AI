def check_overlap(region1, region2):
    """두 영역(bbox)의 겹침 비율을 계산하여 임계값과 비교"""
    x1, y1, x2, y2 = region1
    x3, y3, x4, y4 = region2

    # 겹치는 영역 계산
    overlap_x1 = max(x1, x3)
    overlap_y1 = max(y1, y3)
    overlap_x2 = min(x2, x4)
    overlap_y2 = min(y2, y4)

    # 겹치는 영역이 없으면 False 반환
    if overlap_x2 < overlap_x1 or overlap_y2 < overlap_y1:
        return False

    # 겹치는 영역의 면적
    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    
    # 머리 영역의 면적
    head_area = (x2 - x1) * (y2 - y1)
    
    # 겹치는 비율이 50% 이상이면 True 반환
    return (overlap_area / head_area) >= 0.5