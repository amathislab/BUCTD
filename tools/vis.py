import cv2

def plot_keypoints(image, keypoints, dataset='coco', color='red'):
    """Plot keypoints on the image."""
    # fig, ax = plt.subplots(1, figsize=(12, 9))
    # ax.imshow(image)

    if dataset == 'coco':
        # COCO keypoints order and connections
        keypoints_order = ['nose','left_eye','right_eye','left_ear','right_ear',
                           'left_shoulder','right_shoulder','left_elbow','right_elbow',
                           'left_wrist','right_wrist','left_hip','right_hip',
                           'left_knee','right_knee','left_ankle','right_ankle']

        skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 6), 
                    (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), 
                    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]
    
    elif dataset == 'crowdpose':
        # CrowdPose keypoints order and connections
        keypoints_order = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                           'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 
                           'right_knee', 'left_ankle', 'right_ankle', 'head', 'neck']

        skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 6), (1, 7), 
                    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (0, 12), 
                    (1, 12), (12, 13), (2, 13), (3, 13)]
    else:
        raise ValueError("Dataset not supported. Choose 'coco' or 'crowdpose'")
    
    
    for i, joint in enumerate(keypoints):
        # Draw keypoints
        x, y = joint.astype(int)
        cv2.circle(image, (int(x), int(y)), 3, color, thickness=2)

    # Draw skeleton
    for start, end in skeleton:
        
        x1, y1 = keypoints[start].astype(int)
        x2, y2 = keypoints[end].astype(int)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    return image
