def overlapping_area(detection_1, detection_2):
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[3]
    x2_br = detection_2[0] + detection_2[3]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[4]
    y2_br = detection_2[1] + detection_2[4]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[3] * detection_2[4]
    area_2 = detection_2[3] * detection_2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    if len(detections) == 0:
	   return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[2],
            reverse=True)

    # Unique detections will be appended to this list
    new_detections=[]
    # Append the first detection
    new_detections.append(detections[0])

    # Remove the detection from the original list
    del detections[0]
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            label_det = detection[5]
            label_new_det = detection[5]
            print label_det, " :: ", label_new_det
            if overlapping_area(detection, new_detection) > threshold and label_det == label_new_det:
                del detections[index]
                break
            else:
                new_detections.append(detection)
                # del detections[index]
    return new_detections

if __name__ == "__main__":
    # Example of how to use the NMS Module
    detections = [[31, 31, .9, 10, 10], [31, 31, .12, 10, 10], [100, 34, .8,10, 10]]
    print "Detections before NMS = {}".format(detections)
    print "Detections after NMS = {}".format(nms(detections))
