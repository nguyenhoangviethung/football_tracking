from collections import deque

import cv2
import numpy as np
import supervision as sv

class BallTracker:
    def __init__(self, buffer_size=64):
        self.buffer = deque(maxlen=buffer_size)

    def update(self, detections):
        xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        self.buffer.appendleft(xy)
        
        if len(detections) == 0:
            return detections

        centroid = np.mean(np.concatenate(self.buffer), axis=0)
        distances = np.linalg.norm(xy - centroid, axis=1)
        index = np.argmin(distances)
        return detections[[index]]

class BallAnnotator:
    def __init__(self, radius, buffer_size=64, thinkness=2):
        self.color_pallete = sv.ColorPalette.from_matplotlib('tab10',buffer_size)
        self.buffer = deque(maxlen=buffer_size)
        self.radius = radius
        self.thinkness = thinkness

    def interpolate(self, i, max_i):
        if max_i == 1:
            return self.radius
        return int(1 + i * (self.radius - 1) / (max_i - 1))
    
    def annotate(self, frame, detections):
        xy = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER).astype(int)
        self.buffer.append(xy)
        for i, xy in enumerate(self.buffer):
            color = self.color_palette.by_idx(i)
            interpolated_radius = self.interpolate_radius(i, len(self.buffer))
            for center in xy:
                frame = cv2.circle(
                    img=frame,
                    center=tuple(center),
                    radius=interpolated_radius,
                    color=color.as_bgr(),
                    thickness=self.thickness
                )
        return frame