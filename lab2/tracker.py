"""Tracking a textured planar object using first-frame SIFT correspondences."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    corners: np.ndarray | None = None
    matches: int = 0
    inliers: int = 0
    error: float | None = None
    method: str = "sift"


class PlanarTracker:
    """SIFT relocalization with forward/backward optical-flow fallback."""

    def __init__(self, first_frame, ratio=0.75, min_inliers=10, ransac_threshold=3.0):
        self.ratio = ratio
        self.min_inliers = min_inliers
        self.ransac_threshold = ransac_threshold
        self.sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.025)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        point_sets, descriptor_sets = [], []
        # Small references retain features that disappear when the large template
        # is compared directly with a distant object in a low-resolution video.
        for scale in (1.0, 0.5, 0.25):
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keypoints, descriptors = self.sift.detectAndCompute(scaled, None)
            if descriptors is not None:
                point_sets.append(np.float32([point.pt for point in keypoints]) / scale)
                descriptor_sets.append(descriptors)
        if not descriptor_sets or sum(map(len, point_sets)) < min_inliers:
            raise ValueError("The first frame has too few features; use a textured object.")
        self.descriptors = np.concatenate(descriptor_sets)
        self.points = np.concatenate(point_sets)
        h, w = gray.shape
        self.corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.template_area = float((w - 1) * (h - 1))
        self.previous_gray = gray
        self.flow_source = self.points.copy()
        self.flow_target = self.points.copy()

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = self._match(gray)
        if result.corners is None and len(self.flow_target) >= self.min_inliers:
            previous = self.flow_target.reshape(-1, 1, 2)
            current, status, _ = cv2.calcOpticalFlowPyrLK(
                self.previous_gray, gray, previous, None, winSize=(21, 21), maxLevel=3)
            if current is not None:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.previous_gray, current, None, winSize=(21, 21), maxLevel=3)
                if back is not None:
                    valid = (status.ravel().astype(bool) & back_status.ravel().astype(bool)
                             & (np.linalg.norm(previous[:, 0] - back[:, 0], axis=1) < 1.0))
                    h, w = gray.shape
                    xy = current[:, 0]
                    valid &= ((xy[:, 0] >= 0) & (xy[:, 0] < w)
                              & (xy[:, 1] >= 0) & (xy[:, 1] < h))
                    flow_result = self._estimate(self.flow_source[valid], xy[valid], gray)
                    flow_result.method = "flow"
                    if flow_result.corners is not None:
                        result = flow_result
        self.previous_gray = gray
        if result.corners is None:
            self.flow_source = np.empty((0, 2), np.float32)
            self.flow_target = np.empty((0, 2), np.float32)
        return result

    def _match(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return Detection()
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [pair[0] for pair in pairs
                if len(pair) == 2 and pair[0].distance < self.ratio * pair[1].distance]
        # A scene point must not count as several independent observations.
        unique = {}
        for match in sorted(good, key=lambda item: item.distance):
            unique.setdefault(match.trainIdx, match)
        good = list(unique.values())
        source = self.points[[match.queryIdx for match in good]]
        target = np.float32([keypoints[match.trainIdx].pt for match in good]).reshape(-1, 2)
        result = self._estimate(source, target, gray)
        if result.corners is not None:
            # Refresh well-separated flow points over the object, not the background.
            region = np.zeros_like(gray)
            cv2.fillConvexPoly(region, np.rint(result.corners).astype(np.int32), 255)
            region = cv2.erode(region, np.ones((5, 5), np.uint8))
            points = cv2.goodFeaturesToTrack(gray, 250, 0.01, 4, mask=region)
            if points is not None and len(points) >= self.min_inliers:
                inverse = cv2.getPerspectiveTransform(result.corners, self.corners)
                self.flow_source = cv2.perspectiveTransform(points, inverse)[:, 0]
                self.flow_target = points[:, 0]
        return result

    def _estimate(self, source, target, gray):
        result = Detection(matches=len(source))
        if len(source) < self.min_inliers:
            return result
        matrix, mask = cv2.findHomography(
            source, target, cv2.RANSAC, self.ransac_threshold,
            maxIters=3000, confidence=0.995,
        )
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return result
        mask = mask.ravel().astype(bool)
        result.inliers = int(mask.sum())
        if result.inliers < self.min_inliers or result.inliers / len(source) < 0.4:
            return result
        # Reject a fit supported only by a tiny local patch of the template.
        coverage = cv2.contourArea(cv2.convexHull(source[mask])) / self.template_area
        if coverage < 0.02:
            return result
        homogeneous = np.column_stack((self.corners, np.ones(4))) @ matrix.T
        depths = homogeneous[:, 2]
        if not (np.all(depths > 1e-6) or np.all(depths < -1e-6)):
            return result
        corners = (homogeneous[:, :2] / depths[:, None]).astype(np.float32)
        h, w = gray.shape
        if not np.isfinite(corners).all() or np.abs(corners).max() > 10 * max(h, w):
            return result
        area = cv2.contourArea(corners, oriented=True)
        if not cv2.isContourConvex(corners) or not 100 < area < 4 * h * w:
            return result
        frame_box = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        visible_area, _ = cv2.intersectConvexConvex(corners, frame_box)
        if visible_area < 100:
            return result
        projected = cv2.perspectiveTransform(source[mask, None], matrix)[:, 0]
        result.error = float(np.median(np.linalg.norm(projected - target[mask], axis=1)))
        result.corners = corners
        self.flow_source = source[mask].copy()
        self.flow_target = target[mask].copy()
        return result


def annotate(frame, detection, label):
    """Draw the projected quadrilateral, without modifying the input image."""
    canvas = frame.copy()
    if detection.corners is not None:
        polygon = np.rint(detection.corners).astype(np.int32)
        cv2.polylines(canvas, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        x, y = polygon[np.argmin(polygon[:, 1])]
        x = int(np.clip(x, 0, max(0, frame.shape[1] - 150)))
        y = int(np.clip(y - 8, 22, frame.shape[0] - 8))
        text, color = label, (0, 255, 0)
    else:
        x, y, text, color = 10, 25, "Object not found", (0, 100, 255)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 1, cv2.LINE_AA)
    return canvas
