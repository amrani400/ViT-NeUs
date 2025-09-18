from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
from icecream import ic

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])
    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1) # (batch, N_samples, 2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples

class NeuSRenderer:
    def __init__(self,
                 nerf,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 n_importance,
                 n_outside,
                 up_sample_steps,
                 perturb):
        self.nerf = nerf
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.base_n_importance = n_importance # Base value
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.high_boost = nn.Parameter(torch.ones(color_network.feature_dim) * 1.0) # Per-channel learnable boost
        # Fusion components for low (now optional for SDF, but disabled)
        self.linear_pe_low = nn.Linear(sdf_network.embedded_input_dim, 512)
        self.linear_k_low = nn.Linear(1024, 512)
        self.linear_fused_low = nn.Linear(1024, sdf_network.feature_dim) # 256
        # Fusion components for high (for color)
        self.linear_pe_high = nn.Linear(sdf_network.embedded_input_dim, 512)
        self.linear_k_high = nn.Linear(1024, 512)
        self.linear_fused_high = nn.Linear(1024, color_network.feature_dim) # 256
        # Gate MLP
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def get_n_importance(self, iter_step, end_iter):
        if iter_step < 50000:
            return self.base_n_importance
        else:
            progress = (iter_step - 50000) / (end_iter - 50000)
            return self.base_n_importance + min(int(64 * progress), 32)  

    def fuse_features(self, features_all, pe, C, k, batch_size, n_samples, linear_pe, linear_k, linear_fused, visibility_mask=None):
        features_flat = features_all.reshape(-1, C)
        query = linear_pe(pe)
        keys = linear_k(features_flat)
        values = features_flat
        query_reshaped = query.view(batch_size * n_samples, 1, query.shape[-1])
        keys_reshaped = keys.view(batch_size * n_samples, k, keys.shape[-1])
        attn_logits = torch.bmm(query_reshaped, keys_reshaped.transpose(1, 2))
        attn_logits = attn_logits.view(batch_size, n_samples, k) / (keys.shape[-1] ** 0.5)
        if visibility_mask is not None:
            attn_logits = attn_logits * visibility_mask  # Apply mask to logits as per Eq. (3)
        attn = F.softmax(attn_logits, dim=-1)
        attn_reshaped = attn.view(batch_size * n_samples, k, 1)
        values_reshaped = values.view(batch_size * n_samples, k, C)
        fused = torch.bmm(attn_reshaped.transpose(1, 2), values_reshaped)
        fused = fused.view(batch_size, n_samples, C)
        fused_proj = linear_fused(fused)
        return fused_proj

    def sample_multi_view_features(self, pts, rays_d, feature_maps, poses, intrinsics, padded_sizes, batch_size, n_samples, device, chunk_size=8192):
        pts = pts.reshape(-1, 3)
        k = len(feature_maps)
        features_all = []
        visibility_masks = []  # Collect masks separately
        num_points = batch_size * n_samples
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            pts_chunk = pts[start:end]
            pts_hom_chunk = torch.cat([pts_chunk, torch.ones(end - start, 1, device=device)], dim=-1)
            pts_hom_chunk = pts_hom_chunk.unsqueeze(0).expand(k, -1, -1)
            pts_cam_chunk = torch.einsum('kij,kmj->kmi', torch.inverse(poses), pts_hom_chunk)[..., :3]
            depth_chunk = pts_cam_chunk[..., 2].unsqueeze(-1)  # Depth in camera space
            visibility_mask_chunk = (depth_chunk > 0).float()
            u_chunk = (intrinsics[:, 0, 0].view(k, 1) * pts_cam_chunk[..., 0] / (pts_cam_chunk[..., 2] + 1e-6)) + intrinsics[:, 0, 2].view(k, 1)
            v_chunk = (intrinsics[:, 1, 1].view(k, 1) * pts_cam_chunk[..., 1] / (pts_cam_chunk[..., 2] + 1e-6)) + intrinsics[:, 1, 2].view(k, 1)
            h_padded = torch.tensor([p[0] for p in padded_sizes], device=device).view(k, 1)
            w_padded = torch.tensor([p[1] for p in padded_sizes], device=device).view(k, 1)
            x_grid_chunk = 2 * (u_chunk / (w_padded - 1)) - 1
            y_grid_chunk = 2 * (v_chunk / (h_padded - 1)) - 1
            chunk_features = []
            for i in range(k):
                grid = torch.stack([x_grid_chunk[i], y_grid_chunk[i]], dim=-1).view(1, -1, 1, 2)
                features = F.grid_sample(feature_maps[i].unsqueeze(0), grid, align_corners=True).squeeze(3).squeeze(0).transpose(0,1)
                features = features * visibility_mask_chunk[i]
                chunk_features.append(features)
            chunk_features_all = torch.stack(chunk_features, dim=1)
            features_all.append(chunk_features_all)
            visibility_masks.append(visibility_mask_chunk.view(k, end - start, 1))
        features_all = torch.cat(features_all, dim=0).reshape(batch_size, n_samples, k, -1)
        visibility_masks = torch.cat(visibility_masks, dim=1).reshape(batch_size, n_samples, k, 1)  # Shape for mask
        return features_all, visibility_masks.squeeze(-1)  # Return features and masks

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)
        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)
        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)
        density, sampled_color = nerf(pts, dirs)
        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))
        return {
            'color': color,
            'sampled_color': sampled_color,
            'alpha': alpha,
            'weights': weights,
        }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        prev_cos_val = torch.cat([torch.zeros([batch_size, 1]), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def render_core_coarse(self, rays_o, rays_d, z_vals, sample_dist, sdf_network, deviation_network, feature_maps_low, poses, intrinsics, padded_sizes, cos_anneal_ratio=0.0):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        device = pts.device
        features_all_low, visibility_masks_low = checkpoint(lambda *args: self.sample_multi_view_features(*args), pts.reshape(batch_size, n_samples, 3), dirs, feature_maps_low, poses, intrinsics, padded_sizes, batch_size, n_samples, device)
        pe = sdf_network.embed_fn_fine(pts).reshape(batch_size, n_samples, -1)
        fused_low = checkpoint(lambda *args: self.fuse_features(*args), features_all_low, pe, features_all_low.shape[-1], len(feature_maps_low), batch_size, n_samples, self.linear_pe_low, self.linear_k_low, self.linear_fused_low, visibility_masks_low)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        true_cos = (dirs * sdf_network.gradient(pts).squeeze()).sum(-1, keepdim=True)
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio)
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        entropy = - (weights * torch.log(weights + 1e-10)).sum(dim=-1) / np.log(n_samples)
        gate = self.gate_mlp(entropy.unsqueeze(-1))
        fused_low_gated = fused_low * gate.unsqueeze(1).unsqueeze(2)
        return fused_low_gated, weights, entropy

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    sdf_network,
                    deviation_network,
                    color_network,
                    feature_maps_low,
                    feature_maps_high,
                    poses,
                    intrinsics,
                    padded_sizes,
                    background_alpha=None,
                    background_sampled_color=None,
                    background_rgb=None,
                    cos_anneal_ratio=0.0,
                    n_samples_total=None):  
        batch_size, n_samples = z_vals.shape
        if n_samples_total is None:
            n_samples_total = n_samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        device = pts.device
        features_all_low, visibility_masks_low = checkpoint(lambda *args: self.sample_multi_view_features(*args), pts.reshape(batch_size, n_samples, 3), dirs, feature_maps_low, poses, intrinsics, padded_sizes, batch_size, n_samples, device)
        pe = sdf_network.embed_fn_fine(pts).reshape(batch_size, n_samples, -1)
        fused_low = checkpoint(lambda *args: self.fuse_features(*args), features_all_low, pe, features_all_low.shape[-1], len(feature_maps_low), batch_size, n_samples, self.linear_pe_low, self.linear_k_low, self.linear_fused_low, visibility_masks_low)
        features_all_high, visibility_masks_high = checkpoint(lambda *args: self.sample_multi_view_features(*args), pts.reshape(batch_size, n_samples, 3), dirs, feature_maps_high, poses, intrinsics, padded_sizes, batch_size, n_samples, device)
        fused_high = checkpoint(lambda *args: self.fuse_features(*args), features_all_high, pe, features_all_high.shape[-1], len(feature_maps_high), batch_size, n_samples, self.linear_pe_high, self.linear_k_high, self.linear_fused_high, visibility_masks_high)
        fused_low_flat = fused_low.reshape(-1, fused_low.shape[-1])
        sdf_nn_output = sdf_network(pts, features=fused_low_flat) 
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = sdf_network.gradient(pts, features=fused_low_flat).squeeze()
        fused_high_flat = fused_high.reshape(-1, fused_high.shape[-1])
        sampled_color = color_network(pts, gradients, dirs, feature_vector, fused_high_flat).reshape(batch_size, n_samples, 3)
        inv_s = deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples_total, 1) # Use total samples
        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)
        grad_norm = torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1)
        curvature_error = grad_norm.var(dim=-1).mean() * 0.01
        tv_loss = torch.mean(torch.abs(sdf.reshape(batch_size, n_samples)[:, 1:] - sdf.reshape(batch_size, n_samples)[:, :-1]))
        gradient_error = gradient_error + curvature_error + tv_loss * 0.001
        # High boost
        fused_high = fused_high * self.high_boost
        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, feature_maps_low, feature_maps_high, poses, intrinsics, padded_sizes, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0, iter_step=0, end_iter=300000):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=rays_o.device)
        perturb = self.perturb if perturb_overwrite < 0 else perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_o.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]], device=rays_o.device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand
        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples
        background_alpha = None
        background_sampled_color = None
        n_importance = self.get_n_importance(iter_step, end_iter)
        n_samples_total = self.n_samples + n_importance
        # Coarse pass for gating
        fused_low_gated, coarse_weights, coarse_entropy = self.render_core_coarse(rays_o, rays_d, z_vals, sample_dist, self.sdf_network, self.deviation_network, feature_maps_low, poses, intrinsics, padded_sizes, cos_anneal_ratio)
        if n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, self.n_samples)
                step_size = n_importance // self.up_sample_steps
                remainder = n_importance % self.up_sample_steps
                for i in range(self.up_sample_steps):
                    current_step_size = step_size + (1 if i < remainder else 0)
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                current_step_size,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  last=(i + 1 == self.up_sample_steps))
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)
            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    feature_maps_low,
                                    feature_maps_high,
                                    poses,
                                    intrinsics,
                                    padded_sizes,
                                    background_rgb=background_rgb,
                                    background_alpha=background_alpha,
                                    background_sampled_color=background_sampled_color,
                                    cos_anneal_ratio=cos_anneal_ratio,
                                    n_samples_total=n_samples_total)
        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples_total).mean(dim=-1, keepdim=True)

        return {
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0):
        return extract_geometry(bound_min,
                                bound_max,
                                resolution=resolution,
                                threshold=threshold,
                                query_func=lambda pts: -self.sdf_network.sdf(pts))