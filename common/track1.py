import numpy as np
import supervision as sv
from collections import deque
import cv2
from tqdm import tqdm
from common.ball import BallTracker, BallAnnotator

class FootballTracker:
    def __init__(self, team_classifier, player_model, ball_model, 
                 player_id=2, gk_id=1, ball_id=0,
                 reclassify_interval=30, kit_timeout=60):
 
        self.team_classifier = team_classifier
        self.player_model = player_model
        self.ball_model = ball_model
        
        # ID cấu hình
        self.PLAYER_ID = player_id
        self.GOALKEEPER_ID = gk_id
        self.BALL_ID = ball_id
        
        # Logic cấu hình
        self.reclassify_interval = reclassify_interval
        self.kit_timeout = kit_timeout
        
        # Trackers
        self.global_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=30)
        self.ball_byte_tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=10)
        self.ball_smoother = BallTracker(buffer_size=10) # Thêm bộ làm mượt bóng
        
        # Bộ nhớ (Caches)
        self.team_cache = {}
        self.kit_number_cache = {}
        self.last_seen_cache = {}
        
        # Annotators
        self.team_colors = sv.ColorPalette.from_hex(['#FFD700', '#00BFFF'])
        self.player_annotator = sv.EllipseAnnotator(color=self.team_colors, thickness=2)
        self.player_label_annotator = sv.LabelAnnotator(
            color=self.team_colors, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER
        )
        self.gk_box_annotator = sv.BoxAnnotator(color=self.team_colors, thickness=2)
        
        # Thay thế TriangleAnnotator bằng BallAnnotator xịn sò hơn
        self.ball_annotator = BallAnnotator(radius=12, buffer_size=10, thickness=2)
        # Giữ lại Triangle để đánh dấu vị trí hiện tại cho rõ
        self.ball_marker = sv.TriangleAnnotator(color=sv.Color.WHITE, base=15, height=12)

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
        video_info = sv.VideoInfo.from_video_path(source_path)
        frame_generator = sv.get_video_frames_generator(source_path)

        with sv.VideoSink(target_path, video_info=video_info) as sink:
            for idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):
                
                # --- A. PLAYER TRACKING ---
                p_res = self.player_model.infer(frame, confidence=0.3)[0]
                detections = sv.Detections.from_inference(p_res)
                mask = np.isin(detections.class_id, [self.PLAYER_ID, self.GOALKEEPER_ID])
                detections = detections[mask]
                
                tracked = self.global_tracker.update_with_detections(detections)

                # --- B. TEAM LOGIC ---
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

                # --- C. ATTRIBUTES ---
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

                # --- D. RESOLVE GK ---
                gk_mask = (tracked.class_id == -1)
                if np.any(gk_mask) and p_xy:
                    p_xy, p_team = np.array(p_xy), np.array(p_team)
                    c0 = p_xy[p_team == 0].mean(axis=0) if 0 in p_team else np.array([-9999, -9999])
                    c1 = p_xy[p_team == 1].mean(axis=0) if 1 in p_team else np.array([9999, 9999])
                    
                    for g_idx in np.where(gk_mask)[0]:
                        g_xy = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[g_idx]
                        tracked.class_id[g_idx] = 0 if np.linalg.norm(g_xy-c0) < np.linalg.norm(g_xy-c1) else 1

                # --- E. BALL TRACKING (NÂNG CẤP) ---
                b_res = self.ball_model.infer(frame, confidence=0.09)[0]
                ball_det = sv.Detections.from_inference(b_res)
                # Lọc class Ball (0)
                ball_det = ball_det[ball_det.class_id == self.BALL_ID]
                
                # 1. ByteTrack để duy trì ID
                ball_tracked = self.ball_byte_tracker.update_with_detections(ball_det)
                
                # 2. BallSmoother để làm mượt quỹ đạo
                ball_tracked = self.ball_smoother.update(ball_tracked)

                # --- F. RENDERING ---
                annotated = frame.copy()
                
                # Vẽ Ball Trace (Đuôi bóng) trước để nó nằm dưới các layer khác
                annotated = self.ball_annotator.annotate(annotated, ball_tracked)
                
                # Vẽ Players
                is_gk = np.array([lbl == "GK" for lbl in final_labels])
                annotated = self.player_annotator.annotate(annotated, tracked[~is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[~is_gk], [l for l, g in zip(final_labels, is_gk) if not g])
                
                # Vẽ GK
                annotated = self.gk_box_annotator.annotate(annotated, tracked[is_gk])
                annotated = self.player_label_annotator.annotate(annotated, tracked[is_gk], ["GK"]*sum(is_gk))
                
                # Vẽ Marker cho bóng (Tam giác nhỏ trên đầu)
                if len(ball_tracked) > 0:
                    annotated = self.ball_marker.annotate(annotated, ball_tracked)

                sink.write_frame(annotated)