import numpy as np
import supervision as sv
from scipy.spatial.distance import cdist

# class SmartPlayerTracker:
#     def __init__(self, max_players=22, max_distance_threshold=100, lost_buffer=60):
    
#         self.tracker = sv.ByteTrack(
#             track_activation_threshold=0.25,
#             lost_track_buffer=lost_buffer, 
#             minimum_matching_threshold=0.8,
#             frame_rate=30,
#             minimum_consecutive_frames=3
#         )
        
#         self.max_players = max_players
#         self.max_distance = max_distance_threshold #
        
#         # Ví dụ: {105: 5, 108: 7}
#         self.byte_to_real_map = {} 
        
#         self.last_known_positions = {}
        
#         self.active_real_ids = set()

#     def update(self, detections: sv.Detections) -> sv.Detections:
#         tracked_detections = self.tracker.update_with_detections(detections)
        
#         if len(tracked_detections) == 0:
#             return tracked_detections

#         current_centers = tracked_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         current_byte_ids = tracked_detections.tracker_id
        
#         self.active_real_ids = set()
        
#         new_byte_ids_indices = []

#         for i, byte_id in enumerate(current_byte_ids):
#             if byte_id in self.byte_to_real_map:
#                 real_id = self.byte_to_real_map[byte_id]
#                 self.active_real_ids.add(real_id)
#                 self.last_known_positions[real_id] = current_centers[i]
#             else:
#                 new_byte_ids_indices.append(i)

#         if len(new_byte_ids_indices) > 0:
#             # Tìm các Real ID đang bị thiếu (missing)
#             all_possible_ids = set(range(1, self.max_players + 2)) 
#             missing_ids = list(all_possible_ids - self.active_real_ids)
            
#             candidates = [mid for mid in missing_ids if mid in self.last_known_positions]
            
#             if len(candidates) > 0:

#                 new_track_centers = current_centers[new_byte_ids_indices]
                
#                 candidate_centers = np.array([self.last_known_positions[mid] for mid in candidates])
                

#                 cost_matrix = cdist(new_track_centers, candidate_centers)
                
#                 for row_idx, track_idx_in_detections in enumerate(new_byte_ids_indices):
#                     byte_id = current_byte_ids[track_idx_in_detections]
                    
#                     min_dist_idx = np.argmin(cost_matrix[row_idx])
#                     min_dist = cost_matrix[row_idx][min_dist_idx]
                    
#                     if min_dist < self.max_distance:
#                         matched_real_id = candidates[min_dist_idx]
                        
#                         self.byte_to_real_map[byte_id] = matched_real_id
#                         self.last_known_positions[matched_real_id] = new_track_centers[row_idx]
#                         self.active_real_ids.add(matched_real_id)
                        
#                         cost_matrix[:, min_dist_idx] = 1e9
#                     else:
                        
#                         self._assign_new_id(byte_id, all_possible_ids)
#             else:
        
#                 for idx in new_byte_ids_indices:
#                     self._assign_new_id(current_byte_ids[idx], all_possible_ids)


#         final_ids = []
#         for byte_id in current_byte_ids:
#             if byte_id in self.byte_to_real_map:
#                 final_ids.append(self.byte_to_real_map[byte_id])
#             else:

#                 final_ids.append(byte_id)

#         tracked_detections.tracker_id = np.array(final_ids)
        
#         return tracked_detections

#     def _assign_new_id(self, byte_id, all_possible_ids):
#         available = sorted(list(all_possible_ids - self.active_real_ids))
#         if len(available) > 0:
#             new_real_id = available[0]
#             self.byte_to_real_map[byte_id] = new_real_id
#             self.active_real_ids.add(new_real_id)

class SmartPlayerTracker:
    def __init__(self, max_id=20, max_distance=150):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60, 
            minimum_matching_threshold=0.8,
            frame_rate=30,
            minimum_consecutive_frames=3
        )
        
        self.max_id = max_id
        self.max_distance = max_distance
        

        self.byte_to_real = {}
        
        self.history_positions = {}
        
        self.active_ids = set()

    def update(self, detections: sv.Detections) -> sv.Detections:
        tracked_dets = self.tracker.update_with_detections(detections)
        
        if len(tracked_dets) == 0:
            return tracked_dets

        current_centers = tracked_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        current_byte_ids = tracked_dets.tracker_id
        self.active_ids = set()
        
        unmapped_indices = []

        for i, byte_id in enumerate(current_byte_ids):
            if byte_id in self.byte_to_real:
                real_id = self.byte_to_real[byte_id]
                self.active_ids.add(real_id)
                self.history_positions[real_id] = current_centers[i]
            else:
                unmapped_indices.append(i)

        if len(unmapped_indices) > 0:
          
            all_possible = set(range(1, self.max_id + 1))
            missing_ids = list(all_possible - self.active_ids)
           
            candidates = [mid for mid in missing_ids if mid in self.history_positions]

            if len(candidates) > 0:
                new_centers = current_centers[unmapped_indices]
                cand_centers = np.array([self.history_positions[c] for c in candidates])
                
                dists = cdist(new_centers, cand_centers)
                
                processed_indices = []
                for row_idx, real_idx_in_unmapped in enumerate(unmapped_indices):
                    min_col = np.argmin(dists[row_idx])
                    min_val = dists[row_idx][min_col]
                    
                    if min_val < self.max_distance:
                        byte_id = current_byte_ids[real_idx_in_unmapped]
                        real_id = candidates[min_col]
                        
                        self.byte_to_real[byte_id] = real_id
                        self.history_positions[real_id] = new_centers[row_idx]
                        self.active_ids.add(real_id)
                        
                        dists[:, min_col] = 1e9 
                        processed_indices.append(real_idx_in_unmapped)
                
                unmapped_indices = [idx for idx in unmapped_indices if idx not in processed_indices]

        for idx in unmapped_indices:
            byte_id = current_byte_ids[idx]
            all_possible = set(range(1, self.max_id + 1))
            available = sorted(list(all_possible - self.active_ids))
            
            if len(available) > 0:
                new_id = available[0]
                self.byte_to_real[byte_id] = new_id
                self.active_ids.add(new_id)
                self.history_positions[new_id] = current_centers[idx]
            else:
                self.byte_to_real[byte_id] = 99 

        final_ids = []
        for byte_id in current_byte_ids:
            final_ids.append(self.byte_to_real.get(byte_id, byte_id))
        
        tracked_dets.tracker_id = np.array(final_ids)
        return tracked_dets
