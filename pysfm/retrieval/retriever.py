from pysfm import *
import json
from .netvlad import NetVLAD
from pysfm.utils.imageio import read_image


class ImageRetrieval:
    def __init__(self, model_name='netvlad', device='cpu', save_dir=None) -> None:
        if model_name == 'netvlad':
            self.model = NetVLAD(device)
        
        if save_dir is not None:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, query_image_paths, database_image_paths):
        database_descriptors = []
        for database_image_path in database_image_paths:
            database_image = read_image(database_image_path)[0]
            desc = self.model(database_image).detach()
            database_descriptors.append(desc)
        
        database_descriptors = torch.cat(database_descriptors, dim=0)
        
        query_descriptors = []
        for query_image_path in query_image_paths:
            query_image = read_image(query_image_path)[0]
            desc = self.model(query_image).detach()
            query_descriptors.append(desc)

        query_descriptors = torch.cat(query_descriptors, dim=0)

        sim_matrix = torch.einsum("id, jd -> ij", query_descriptors, database_descriptors)

        topk_dbs = torch.topk(sim_matrix, 3, dim=-1)
        values = topk_dbs.values.cpu().numpy()
        indices = topk_dbs.indices.cpu().numpy()
        
        matches = {}
        for q_idx in range(len(values)):
            value, index = values[q_idx].tolist(), indices[q_idx].tolist()
            matched_paths = [str(database_image_paths[d_idx]) for d_idx in index]
            matches[str(query_image_paths[q_idx])] = matched_paths

        if self.save_dir is not None:
            with open(self.save_dir / "pairs.json", "w") as f:
                json.dump(matches, f)