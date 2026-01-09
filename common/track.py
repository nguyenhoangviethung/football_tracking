from tqdm import tqdm
import numpy as np
import supervision as sv

from common.player import SmartPlayerTracker

# Trackers
player_tracker = SmartPlayerTracker(max_id=20, max_distance=150)
gk_tracker = sv.ByteTrack(track_activation_threshold=0.2, lost_track_buffer=30)
ball_tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=10) # Track bóng để mượt hơn

# Annotators
colors = sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700'])
ellipse_annotator = sv.EllipseAnnotator(color=colors, thickness=2)
label_annotator = sv.LabelAnnotator(color=colors, text_color=sv.Color.BLACK, text_position=sv.Position.BOTTOM_CENTER)
gk_box_annotator = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
gk_label_annotator = sv.LabelAnnotator(color=sv.Color.RED, text_color=sv.Color.WHITE)
ref_box_annotator = sv.BoxAnnotator(color=sv.Color.BLACK, thickness=2)
triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FFD700'), base=25, height=21, outline_thickness=1)

# ==========================================
# 3. VÒNG LẶP XỬ LÝ VIDEO
# ==========================================
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
TARGET_VIDEO_PATH = "/content/output_tracked_final.mp4"
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

print(f"Processing video: {SOURCE_VIDEO_PATH} -> {TARGET_VIDEO_PATH}")

with sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info) as sink:
    # Dùng tqdm để hiện thanh tiến trình
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        # --- A. INFERENCE ---
        # 1. Detect Người (Player Model)
        p_result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        all_detections = sv.Detections.from_inference(p_result)

        # 2. Detect Bóng (Ball Model)
        b_result = BALL_DETECTION_MODEL.infer(frame, confidence=0.25)[0] # Giảm conf chút cho bóng
        ball_detections = sv.Detections.from_inference(b_result)
        ball_detections = ball_detections[ball_detections.class_id == 0] # Chỉ lấy bóng
        # Pad box bóng to ra chút để vẽ tam giác đẹp hơn
        if len(ball_detections) > 0:
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # --- B. TRACKING & FILTERING ---

        # 1. Xử lý Players (Class 2)
        players = all_detections[all_detections.class_id == 2]
        players = players.with_nms(threshold=0.5, class_agnostic=True)
        tracked_players = player_tracker.update(players) # -> ID 1-22

        # 2. Xử lý Goalkeepers (Class 1)
        gks = all_detections[all_detections.class_id == 1]
        tracked_gks = gk_tracker.update_with_detections(gks)
        # Ép ID về 0
        if len(tracked_gks) > 0:
            tracked_gks.tracker_id = np.zeros_like(tracked_gks.tracker_id)

        # 3. Xử lý Referee (Class 3)
        refs = all_detections[all_detections.class_id == 3]
        # Không cần track ID, chỉ cần detections để vẽ box

        # 4. Xử lý Bóng (Class 0 từ Model 2)
        tracked_ball = ball_tracker.update_with_detections(ball_detections)

        # --- C. DRAWING ---
        annotated_frame = frame.copy()

        # Vẽ Players
        labels_player = [f"#{tid}" for tid in tracked_players.tracker_id]
        annotated_frame = ellipse_annotator.annotate(annotated_frame, tracked_players)
        annotated_frame = label_annotator.annotate(annotated_frame, tracked_players, labels_player)

        # Vẽ Goalkeepers
        labels_gk = ["GK" for _ in tracked_gks.tracker_id]
        annotated_frame = gk_box_annotator.annotate(annotated_frame, tracked_gks)
        annotated_frame = gk_label_annotator.annotate(annotated_frame, tracked_gks, labels_gk)

        # Vẽ Referee
        annotated_frame = ref_box_annotator.annotate(annotated_frame, refs)

        # Vẽ Bóng
        annotated_frame = triangle_annotator.annotate(annotated_frame, tracked_ball)

        # Lưu frame
        sink.write_frame(annotated_frame)

print("Hoàn tất! Video đã được lưu.")