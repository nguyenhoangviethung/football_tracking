from typing import Generator, Iterable, List, TypeVar
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")
SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

def create_batches(sequence: Iterable[V], batch_size: int) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch
class TeamClassifier:
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    # THÊM tham số verbose=False mặc định để tắt tqdm khi predict
    def extract_features(self, crops: List[np.ndarray], verbose: bool = False) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        
        # Logic bật/tắt tqdm
        iterator = tqdm(batches, desc='Embedding extraction', leave=False) if verbose else batches
        
        with torch.no_grad():
            for batch in iterator:
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
                
        if not data:
            return np.array([])
        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        # Khi fit (train), ta muốn nhìn thấy tiến độ -> verbose=True
        data = self.extract_features(crops, verbose=True)
        if len(data) == 0:
            return
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])
        # Khi predict từng frame, ta KHÔNG muốn hiện tqdm -> verbose=False (mặc định)
        data = self.extract_features(crops, verbose=False)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

    def resolve_goalkeepers_team_id(self, players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
        goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        
        # Handle trường hợp không có player nào để tránh lỗi mean()
        if len(players_xy) == 0:
             return np.zeros(len(goalkeepers), dtype=int)

        # Lọc toạ độ theo team
        mask_0 = (players.class_id == 0)
        mask_1 = (players.class_id == 1)
        
        # Nếu một trong 2 team không có người, gán tạm centroid xa vô cực hoặc lấy centroid của team kia
        # Để đơn giản, ta check if len > 0
        if np.any(mask_0):
            team_0_centroid = players_xy[mask_0].mean(axis=0)
        else:
            team_0_centroid = np.array([[-99999, -99999]]) # Xa vô tận

        if np.any(mask_1):
            team_1_centroid = players_xy[mask_1].mean(axis=0)
        else:
            team_1_centroid = np.array([[99999, 99999]]) # Xa vô tận
        
        goalkeepers_team_id = []
        for goalkeeper_xy in goalkeepers_xy:
            dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
            dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
            goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

        return np.array(goalkeepers_team_id)