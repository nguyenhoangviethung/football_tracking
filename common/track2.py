import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from collections import deque

# Import các công cụ vẽ từ module của bạn
from annotators.soccer import (
    draw_pitch, 
    draw_points_on_pitch, 
    draw_pitch_voronoi_diagram
)
from configs.config import SoccerPitchConfiguration

# --- Helper Class: ViewTransformer ---
class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # Sử dụng findHomography để ổn định hơn khi có > 4 điểm
        self.m, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self.m is None:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

# --- Main Class: FootballTracker ---
class FootballTracker:
    def __init__(self, team_classifier, player_model, ball_model, field_model=None,
                 player_id=2, gk_id=1, ball_id=0,
                 reclassify_interval=30, kit_timeout=60):
        """
        Args:
            field_model: Model detect sân (cần thiết cho hologram), có thể None nếu chỉ dùng tracking thường.
        """
        self.team_classifier = team_classifier
        self.player_model = player_model
        self.ball_model = ball_model
        self.field_model = field_model  # Thêm model sân
        
        # ID cấu hình
        self.PLAYER_ID = player_id
        self.GOALKEEPER_ID = gk_id
        self.BALL_ID = ball_id
        
        # Logic cấu hình
        self.reclassify_interval = reclassify_interval
        self.kit_timeout = kit_timeout
        
        # Trackers
        self.global_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        # Đổi tên cho rõ ràng, hỗ trợ code cũ dùng ball_tracker
        self.ball_byte_tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=10)
        self.ball_tracker = self.ball_byte_tracker 
        
        # Bộ nhớ (Caches)
        self.team_cache = {}
        self.kit_number_cache = {}
        self.last_seen_cache = {}
        
        # Annotators
        self.team_colors = sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700'])
        self.player_annotator = sv.EllipseAnnotator(color=self.team_colors, thickness=2)
        self.player_label_annotator = sv.LabelAnnotator(
            color=self.team_colors, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER
        )
        self.gk_box_annotator = sv.BoxAnnotator(color=self.team_colors, thickness=2)
        self.ball_annotator = sv.TriangleAnnotator(color=sv.Color.WHITE, base=20, height=17)

    def _assign_kit_number(self, tracker_id, team_id, current_frame_idx):
        self.last_seen_cache[tracker_id] = current_frame_idx
        
        if tracker_id in self.kit_number_cache:
            return self.kit_number_cache[tracker_id]

        taken_numbers = set()
        for tid in list(self.kit_number_cache.keys()):
            if current_frame_idx - self.last_seen_cache.get(tid, 0) > self.kit_timeout:
                del self.kit_number_cache[tid]
                continue
            
            if self.team_cache.get(tid) == team_id:
                taken_numbers.add(self.kit_number_cache[tid])

        for num in range(1, 11):
            if num not in taken_numbers:
                self.kit_number_cache[tracker_id] = num
                return num
        
        return (tracker_id - 1) % 10 + 1

    def process_video(self, source_path, target_path):
        """Hàm tracking cơ bản (Giữ nguyên logic cũ để tham chiếu)"""
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        with sv.VideoSink(target_path, video_info=video_info) as sink:
            for idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
                # A. Detection & Tracking
                p_res = self.player_model.infer(frame, confidence=0.3)[0]
                detections = sv.Detections.from_inference(p_res)
                mask = np.isin(detections.class_id, [self.PLAYER_ID, self.GOALKEEPER_ID])
                detections = detections[mask]
                
                tracked = self.global_tracker.update_with_detections(detections)

                # B. Team Classification
                should_reclassify = (idx % self.reclassify_interval == 0)
                unknown_crops, unknown_idxs = [], []

                for i, (tid, cid) in enumerate(zip(tracked.tracker_id, tracked.class_id)):
                    if cid == self.PLAYER_ID:
                        if tid not in self.team_cache or should_reclassify:
                            unknown_crops.append(sv.crop_image(frame, tracked.xyxy[i]))
                            unknown_idxs.append(i)

                if unknown_crops:
                    new_teams = self.team_classifier.predict(unknown_crops)
                    for u_idx, nt_id in zip(unknown_idxs, new_teams):
                        tid = tracked.tracker_id[u_idx]
                        if self.team_cache.get(tid) is not None and self.team_cache.get(tid) != nt_id:
                            self.kit_number_cache.pop(tid, None)
                        self.team_cache[tid] = nt_id

                # C. Attributes
                final_cids, final_labels, p_xy, p_team = [], [], [], []
                for i, (tid, cid) in enumerate(zip(tracked.tracker_id, tracked.class_id)):
                    if cid == self.PLAYER_ID:
                        t_id = self.team_cache.get(tid, 0)
                        k_num = self._assign_kit_number(tid, t_id, idx)
                        final_cids.append(t_id)
                        final_labels.append(f"#{k_num}")
                        p_xy.append(tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i])
                        p_team.append(t_id)
                    else:
                        final_cids.append(-1)
                        final_labels.append("GK")

                tracked.class_id = np.array(final_cids)

                # D. Resolve GK
                gk_mask = (tracked.class_id == -1)
                if np.any(gk_mask) and p_xy:
                    p_xy_arr, p_team_arr = np.array(p_xy), np.array(p_team)
                    c0 = p_xy_arr[p_team_arr == 0].mean(axis=0) if 0 in p_team_arr else np.array([-9999, -9999])
                    c1 = p_xy_arr[p_team_arr == 1].mean(axis=0) if 1 in p_team_arr else np.array([9999, 9999])
                    
                    for g_idx in np.where(gk_mask)[0]:
                        g_xy = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[g_idx]
                        tracked.class_id[g_idx] = 0 if np.linalg.norm(g_xy-c0) < np.linalg.norm(g_xy-c1) else 1

                # E. Ball & Rendering
                b_res = self.ball_model.infer(frame, confidence=0.25)[0]
                ball_det = sv.Detections.from_inference(b_res)
                ball_tracked = self.ball_byte_tracker.update_with_detections(ball_det[ball_det.class_id == 0])

                is_gk = np.array([lbl == "GK" for lbl in final_labels])
                annotated = self.player_annotator.annotate(frame.copy(), tracked[~is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[~is_gk], [l for l, g in zip(final_labels, is_gk) if not g])
                annotated = self.gk_box_annotator.annotate(annotated, tracked[is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[is_gk], ["GK"]*sum(is_gk))
                if len(ball_tracked) > 0:
                    annotated = self.ball_annotator.annotate(annotated, ball_tracked)

                sink.write_frame(annotated)

    def process_video_with_hologram(self, source_path, target_path, config=SoccerPitchConfiguration()):
        """Hàm mới: Tracking + Hologram Overlay (Picture-in-Picture)"""
        if self.field_model is None:
            print("Error: field_model chưa được cung cấp khi khởi tạo FootballTracker.")
            return

        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)
        
        # Helper resize
        def resize_keep_aspect(img, target_h):
            h, w = img.shape[:2]
            scale = target_h / h
            return cv2.resize(img, (int(w * scale), target_h))

        with sv.VideoSink(target_path, video_info=video_info) as sink:
            for idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
                # --- A. LOGIC TRACKING (COPY TỪ HÀM GỐC) ---
                p_res = self.player_model.infer(frame, confidence=0.3)[0]
                detections = sv.Detections.from_inference(p_res)
                mask = np.isin(detections.class_id, [self.PLAYER_ID, self.GOALKEEPER_ID])
                detections = detections[mask]
                
                tracked = self.global_tracker.update_with_detections(detections)

                # Team Classification
                should_reclassify = (idx % self.reclassify_interval == 0)
                unknown_crops, unknown_idxs = [], []
                for i, (tid, cid) in enumerate(zip(tracked.tracker_id, tracked.class_id)):
                    if cid == self.PLAYER_ID:
                        if tid not in self.team_cache or should_reclassify:
                            unknown_crops.append(sv.crop_image(frame, tracked.xyxy[i]))
                            unknown_idxs.append(i)
                if unknown_crops:
                    new_teams = self.team_classifier.predict(unknown_crops)
                    for u_idx, nt_id in zip(unknown_idxs, new_teams):
                        tid = tracked.tracker_id[u_idx]
                        if self.team_cache.get(tid) is not None and self.team_cache.get(tid) != nt_id:
                            self.kit_number_cache.pop(tid, None)
                        self.team_cache[tid] = nt_id

                # Attributes
                final_cids, final_labels, p_xy, p_team = [], [], [], []
                for i, (tid, cid) in enumerate(zip(tracked.tracker_id, tracked.class_id)):
                    if cid == self.PLAYER_ID:
                        t_id = self.team_cache.get(tid, 0)
                        k_num = self._assign_kit_number(tid, t_id, idx)
                        final_cids.append(t_id)
                        final_labels.append(f"#{k_num}")
                        p_xy.append(tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i])
                        p_team.append(t_id)
                    else:
                        final_cids.append(-1)
                        final_labels.append("GK")
                tracked.class_id = np.array(final_cids)

                # Ball
                b_res = self.ball_model.infer(frame, confidence=0.25)[0]
                ball_det = sv.Detections.from_inference(b_res)
                ball_tracked = self.ball_byte_tracker.update_with_detections(ball_det[ball_det.class_id == 0])

                # --- B. LOGIC HOLOGRAM ---
                f_res = self.field_model.infer(frame, confidence=0.3)[0]
                kpts = sv.KeyPoints.from_inference(f_res)
                kpt_filter = kpts.confidence[0] > 0.5
                
                has_homography = False
                pitch_people_xy = np.empty((0, 2))
                pitch_ball_xy = np.empty((0, 2))

                if np.sum(kpt_filter) >= 4:
                    transformer = ViewTransformer(
                        source=kpts.xy[0][kpt_filter], 
                        target=np.array(config.vertices)[kpt_filter]
                    )
                    
                    if transformer.m is not None:
                        people_xy_all = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                        pitch_people_xy = transformer.transform_points(people_xy_all)
                        
                        if len(ball_tracked) > 0:
                            ball_xy_all = ball_tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                            pitch_ball_xy = transformer.transform_points(ball_xy_all)
                        has_homography = True

                # Resolve GK using best available coords
                gk_mask = (tracked.class_id == -1)
                if np.any(gk_mask) and (len(p_xy) > 0 or len(pitch_people_xy) > 0):
                    coords = pitch_people_xy if (has_homography and len(pitch_people_xy) > 0) else np.array(p_xy)
                    if len(coords) == len(tracked): # Safety check
                        t0_mask, t1_mask = tracked.class_id == 0, tracked.class_id == 1
                        c0 = coords[t0_mask].mean(axis=0) if np.any(t0_mask) else np.array([-9999, -9999])
                        c1 = coords[t1_mask].mean(axis=0) if np.any(t1_mask) else np.array([9999, 9999])
                        
                        for g_idx in np.where(gk_mask)[0]:
                            g_xy = coords[g_idx]
                            tracked.class_id[g_idx] = 0 if np.linalg.norm(g_xy-c0) < np.linalg.norm(g_xy-c1) else 1

                # --- C. RENDERING ---
                annotated = frame.copy()
                is_gk = np.array(["GK" in l for l in final_labels])
                
                # Vẽ Player/GK
                annotated = self.player_annotator.annotate(annotated, tracked[~is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[~is_gk], [l for l, g in zip(final_labels, is_gk) if not g])
                annotated = self.gk_box_annotator.annotate(annotated, tracked[is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[is_gk], ["GK"]*sum(is_gk))
                if len(ball_tracked) > 0:
                    annotated = self.ball_annotator.annotate(annotated, ball_tracked)

                # Vẽ Overlay
                if has_homography:
                    radar = draw_pitch(config)
                    t0 = pitch_people_xy[tracked.class_id == 0]
                    t1 = pitch_people_xy[tracked.class_id == 1]
                    
                    if len(t0) >= 2 and len(t1) >= 2:
                        try:
                            radar = draw_pitch_voronoi_diagram(config, t0, t1, 
                                                               sv.Color.from_hex('#00BFFF'), sv.Color.from_hex('#FF1493'), 
                                                               opacity=0.6, pitch=radar)
                        except: pass
                    
                    radar = draw_points_on_pitch(config, t0, face_color=sv.Color.from_hex('#00BFFF'), radius=10, pitch=radar)
                    radar = draw_points_on_pitch(config, t1, face_color=sv.Color.from_hex('#FF1493'), radius=10, pitch=radar)
                    radar = draw_points_on_pitch(config, pitch_ball_xy, face_color=sv.Color.WHITE, radius=8, pitch=radar)
                    
                    # Overlay logic
                    target_h = int(annotated.shape[0] * 0.3)
                    radar_small = resize_keep_aspect(radar, target_h)
                    h_r, w_r = radar_small.shape[:2]
                    
                    margin = 20
                    y1 = annotated.shape[0] - h_r - margin
                    x1 = annotated.shape[1] - w_r - margin
                    
                    annotated[y1:y1+h_r, x1:x1+w_r] = radar_small
                    cv2.rectangle(annotated, (x1, y1), (x1+w_r, y1+h_r), (255, 255, 255), 2)

                sink.write_frame(annotated)