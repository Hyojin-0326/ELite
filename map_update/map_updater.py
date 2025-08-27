import os
import yaml
import numpy as np
import open3d as o3d
from tqdm import trange
from scipy.spatial import KDTree
import faiss


from utils.session import Session
from utils.session_map import SessionMap
from utils.logger import logger


class MapRemover:
    def __init__(
        self, 
        config_path: str
    ):
        # Load parameters
        with open(config_path, 'r') as f:
            self.params = yaml.safe_load(f)

        p_settings = self.params["settings"]        
        os.makedirs(p_settings["output_dir"], exist_ok=True)
        
        self.std_dev_o = 0.025
        self.std_dev_f = 0.025
        self.alpha = 0.5
        self.beta = 0.1

        self.session_loader : Session = None
        self.session_map : SessionMap = None

        self.faiss_index = None
        self.num_scans = None


    def load(self, new_session : Session = None):
        p_settings = self.params["settings"]

        if new_session is None:
            self.session_loader = Session(p_settings["scans_dir"], p_settings["poses_file"])
        else:
            self.session_loader = new_session

        self.num_scans = len(self.session_loader)
        logger.info(f"Loaded new session with {self.num_scans} scans")
        

    def build_faiss_index(self, anchor_points):
        if isinstance(anchor_points, o3d.geometry.PointCloud):
            anchor_points = np.asarray(anchor_points.points, dtype=np.float32)
        elif isinstance(anchor_points, np.ndarray):
            anchor_points = anchor_points.astype(np.float32)
        else:
            raise TypeError(f"Unsupported anchor_points type: {type(anchor_points)}")
        dim, m = 3, 32
        self.faiss_index = faiss.IndexHNSWFlat(dim, m)
        self.faiss_index.add(anchor_points)
        logger.info(f"Built FAISS HNSW index with {anchor_points.shape[0]} points")

    def faiss_knn(self,queries, k):
        ## d2값인지 d인지 모르겠음 일단 d로해봄
        if isinstance(queries, o3d.geometry.PointCloud):
            queries = np.asarray(queries.points, dtype=np.float32)
        elif isinstance(queries, np.ndarray):
            queries = queries.astype(np.float32)
        else: raise TypeError(f"Unsupported anchor_points type: {type(queries)}")
        d, idx = self.faiss_index.search(queries, k)
        return d, idx
        

    def run(self):
        p_settings = self.params["settings"]
        p_dor = self.params["dynamic_object_removal"]

        # 1) Aggregate scans to create session map
        session_map = self.session_loader[0:self.num_scans].downsample(0.01).get()
        eph_l = np.zeros(len(session_map.points))
        logger.info(f"Initialized session map")

        # 2) Select anchor points for local ephemerality update
        anchor_points = session_map.voxel_down_sample(p_dor["anchor_voxel_size"])
        anchor_eph_l = np.ones(len(anchor_points.points)) * 0.5 # initial value
        self.build_faiss_index(anchor_points)
        odds = anchor_eph_l / (1 - anchor_eph_l)
        

        logger.info(f"Updating anchor local ephemerality")
        for i in trange(0, self.num_scans, p_dor["stride"], desc="Updating \u03B5_l", ncols=100):

            logger.debug(f"Processing scan {i + 1}/{self.num_scans}")
            scan = np.asarray(self.session_loader[i].get().points)
            pose = self.session_loader.get_pose(i)[:3, 3]
            
            # occupied space update
            dist, inds = self.faiss_knn(scan, k=p_dor["num_k"])

                #flatten 
            dists = dists.ravel()
            inds = inds.ravel()
            eph_l_prev_flatten = anchor_eph_l[inds]

            r = np.minimum(
                self.alpha * (1 - np.exp(-dist**2 / self.std_dev_o)) + self.beta,
                self.alpha
            )
            eph_l_new_flatten = eph_l_prev_flatten * update_rate / (
                    eph_l_prev_flatten * update_rate + (1 - eph_l_prev_flatten) * (1 - update_rate)
                )
            
            np.put(anchor_eph_l, inds, eph_l_new_flatten)






            for j in range(len(dists)):
                dist = dists[j]
                eph_l_prev = anchor_eph_l[inds[j]]
                update_rate = np.minimum(self.alpha * (1 - np.exp(-1 * dist**2 / self.std_dev_o)) + self.beta, self.alpha) # Eq. 5 
                eph_l_new = eph_l_prev * update_rate / (
                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                )
                anchor_eph_l[inds[j]] = eph_l_new

            # free space update
            shifted_scan = scan - pose # local coordinates
            sample_ratios = np.linspace(p_dor["min_ratio"], p_dor["max_ratio"], p_dor["num_samples"])
            free_space_samples = pose + shifted_scan[:, np.newaxis, :] * sample_ratios.T[np.newaxis, :, np.newaxis]
            free_space_samples = free_space_samples.reshape(-1, 3)
            free_space_samples_o3d = o3d.geometry.PointCloud()
            free_space_samples_o3d.points = o3d.utility.Vector3dVector(free_space_samples)
            free_space_samples_o3d = free_space_samples_o3d.voxel_down_sample(voxel_size=0.1)
            free_space_samples = np.asarray(free_space_samples_o3d.points)
            dists, inds = anchor_kdtree.query(free_space_samples, k=p_dor["num_k"])
            for j in range(len(dists)):
                dist = dists[j]
                eph_l_prev = anchor_eph_l[inds[j]]
                update_rate = np.maximum(self.alpha * (1 + np.exp(-1 * dist**2 / self.std_dev_f)) - self.beta, self.alpha) # Eq. 5
                eph_l_new = eph_l_prev * update_rate / (
                    eph_l_prev * update_rate + (1 - eph_l_prev) * (1 - update_rate)
                )
                anchor_eph_l[inds[j]] = eph_l_new

        # 3) Propagate anchor local ephemerality to session map
        distances, indices = anchor_kdtree.query(np.asarray(session_map.points), k=p_dor["num_k"])
        distances = np.maximum(distances, 1e-6) # avoid division by zero
        weights = 1 / (distances**2)
        weights /= np.sum(weights, axis=1, keepdims=True)
        eph_l = np.sum(weights * anchor_eph_l[indices], axis=1)
        eph_l = np.clip(eph_l, 0, 1) # redundant, but for safety

        # 4) Remove dynamic objects to create cleaned session map
        static_points = session_map.select_by_index(np.where(eph_l <= p_dor["dynamic_threshold"])[0])
        static_eph_l = eph_l[eph_l <= p_dor["dynamic_threshold"]]
        static_points.paint_uniform_color([0.5, 0.5, 0.5])
        dynamic_points = session_map.select_by_index(np.where(eph_l > p_dor["dynamic_threshold"])[0])
        dynamic_points.paint_uniform_color([1, 0, 0])
                  
        if p_dor["save_static_dynamic_map"]:
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "static_points.pcd"), static_points)  
            o3d.io.write_point_cloud(os.path.join(p_settings["output_dir"], "dynamic_points.pcd"), dynamic_points)
        if p_dor["viz_static_dynamic_map"]:
            total_points = static_points + dynamic_points
            o3d.visualization.draw_geometries([total_points])

        cleaned_session_map = SessionMap(
            np.asarray(static_points.points), static_eph_l
        )
        self.session_map = cleaned_session_map

        if p_dor["save_cleaned_session_map"]:
            cleaned_session_map.save(p_settings["output_dir"], is_global=False) 
        if p_dor["viz_cleaned_session_map"]:
            cleaned_session_map.visualize()

        return cleaned_session_map


    def get(self):
        return self.session_map
        

# Example usage
if __name__ == "__main__":
    config = "../config/sample.yaml"
    remover = MapRemover(config)
    # Load session using the config file or from an alingment module
    remover.load()
    # Run the dynamic object removal
    remover.run()
    # Get the cleaned session map
    cleaned_session_map = remover.get()