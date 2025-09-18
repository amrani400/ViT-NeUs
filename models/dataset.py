import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from .vit_feature_extractor import ViTFeatureExtractor
from torchvision import transforms
from scipy.spatial import cKDTree

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()
    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]
    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]
    return intrinsics, pose

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.scale_mats_np = []
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W
        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01, 1.01, 1.01, 1.0])
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]
        self.vit_extractor = ViTFeatureExtractor(vit_size='vitl', patch_size=14, layer_to_extract=23)
        self.compute_freq_feature_maps()
        self.compute_neighboring_views_dynamic(k=3, num_sample_points=100) 
        print('Load data: End')

    def compute_freq_feature_maps(self):
        self.vit_extractor.eval()
        self.vit_extractor.to(self.device)
        self.feature_maps_low = []
        self.feature_maps_high = []
        self.padded_sizes = []
        with torch.no_grad():
            for img in self.images:
                img = img.permute(2, 0, 1).unsqueeze(0).to(self.device)  
                low, high, padded_size = self.vit_extractor(img)  
                self.feature_maps_low.append(low.to(self.device))
                self.feature_maps_high.append(high.to(self.device))
                self.padded_sizes.append(padded_size)

    def compute_neighboring_views_dynamic(self, k=3, num_sample_points=100):
        points = np.random.uniform(self.object_bbox_min, self.object_bbox_max, (num_sample_points, 3))
        points = torch.from_numpy(points).float().to(self.device)
        overlap_matrix = np.zeros((self.n_images, self.n_images))
        for i in range(self.n_images):
            for j in range(self.n_images):
                if i == j:
                    continue
                pose_i = self.pose_all[i]
                intr_i = self.intrinsics_all[i]
                pts_hom = torch.cat([points, torch.ones(num_sample_points, 1, device=self.device)], dim=-1)
                pts_cam_i = torch.einsum('ij,mj->mi', torch.inverse(pose_i), pts_hom)[:, :3]
                visible_i = (pts_cam_i[:, 2] > 0)
                u_i = (intr_i[0, 0] * pts_cam_i[:, 0] / (pts_cam_i[:, 2] + 1e-6)) + intr_i[0, 2]
                v_i = (intr_i[1, 1] * pts_cam_i[:, 1] / (pts_cam_i[:, 2] + 1e-6)) + intr_i[1, 2]
                in_bounds_i = (u_i >= 0) & (u_i < self.W) & (v_i >= 0) & (v_i < self.H)
                visible_i = visible_i & in_bounds_i
                pose_j = self.pose_all[j]
                intr_j = self.intrinsics_all[j]
                pts_cam_j = torch.einsum('ij,mj->mi', torch.inverse(pose_j), pts_hom)[:, :3]
                visible_j = (pts_cam_j[:, 2] > 0)
                u_j = (intr_j[0, 0] * pts_cam_j[:, 0] / (pts_cam_j[:, 2] + 1e-6)) + intr_j[0, 2]
                v_j = (intr_j[1, 1] * pts_cam_j[:, 1] / (pts_cam_j[:, 2] + 1e-6)) + intr_j[1, 2]
                in_bounds_j = (u_j >= 0) & (u_j < self.W) & (v_j >= 0) & (v_j < self.H)
                visible_j = visible_j & in_bounds_j
                shared = visible_i & visible_j
                overlap = shared.sum().item() / max(visible_i.sum().item(), 1)
                overlap_matrix[i, j] = overlap
        neighbor_indices = []
        for i in range(self.n_images):
            scores = overlap_matrix[i]
            indices = np.argsort(-scores)[:k]  
            neighbor_indices.append(indices)
        self.neighbor_indices = np.array(neighbor_indices)

    def gen_rays_at(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size], device='cpu')
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size], device='cpu')
        color = self.images[img_idx][(pixels_y, pixels_x)] # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)] # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float() # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3].cpu(), p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3].cpu(), rays_v[:, :, None]).squeeze() # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].cpu().expand(rays_v.shape) # batch_size, 3
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).to(self.device) # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l, device=self.device)
        ty = torch.linspace(0, self.H - 1, self.H // l, device=self.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze() # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True) # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).to(self.device)
        trans = torch.from_numpy(pose[:3, 3]).to(self.device)
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze() # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape) # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)