import torch 
import torch.nn as nn
import torch.nn.functional as F 

import numpy as np

from backbone import *
from utils import *
from roi_align.roi_align import RoIAlign      # RoIAlign module
from roi_align.roi_align import CropAndResize # crop_and_resize module


class Appearance_GCN_Module(nn.Module):
    """
    Appearance GCN Module as described in the paper
    Adjacency matrix: a_t^ij = f_p(v_t^i, v_t^j) * exp(f_a(v_t^i, v_t^j)) / sum(...)
    where f_a(v_t^i, v_t^j) = θ(v_t^i)^T * φ(v_t^j) / √D
    and f_p is indicator function for tracklet presence
    """
    def __init__(self, cfg):
        super(Appearance_GCN_Module, self).__init__()
        
        self.cfg = cfg
        NFG = cfg.num_features_gcn
        D = NFG  # Dimension of features
        
        # θ and φ projection matrices (w and w' in paper)
        self.fc_theta = nn.Linear(D, D)  # θ(v_t^i) = w * v_t^i
        self.fc_phi = nn.Linear(D, D)    # φ(v_t^j) = w' * v_t^j
        
        # GCN weight matrix W^(l)
        self.fc_gcn = nn.Linear(D, D, bias=False)
        
        # Normalization factor √D
        self.sqrt_D = np.sqrt(D)
        
    def forward(self, appearance_features, boxes_in_flat, presence_mask=None):
        """
        Args:
            appearance_features: [B*T, N, D] appearance features
            boxes_in_flat: [B*T*N, 4] bounding boxes
            presence_mask: [B*T, N] indicator for present tracklets (optional)
        Returns:
            output_features: [B*T, N, D] GCN output features
            adjacency_matrix: [B*T, N, N] adjacency matrix
        """
        B_T, N, D = appearance_features.shape
        
        # Compute θ and φ projections
        theta = self.fc_theta(appearance_features)  # [B*T, N, D]
        phi = self.fc_phi(appearance_features)      # [B*T, N, D]
        
        # Compute f_a(v_t^i, v_t^j) = θ^T * φ / √D
        # theta: [B*T, N, D], phi: [B*T, N, D]
        f_a = torch.matmul(theta, phi.transpose(1, 2)) / self.sqrt_D  # [B*T, N, N]
        
        # Compute f_p: indicator function for presence
        if presence_mask is None:
            # If no explicit mask, assume all boxes with valid coordinates are present
            # Create mask based on boxes: non-zero boxes are present
            boxes_reshaped = boxes_in_flat.reshape(B_T, N, 4)  # [B*T, N, 4]
            presence_mask = (boxes_reshaped.abs().sum(dim=2) > 1e-6).float()  # [B*T, N]
        
        # f_p(v_t^i, v_t^j) = I(v_t^i present AND v_t^j present)
        f_p = presence_mask.unsqueeze(2) * presence_mask.unsqueeze(1)  # [B*T, N, N]
        
        # Compute a_t^ij = f_p * exp(f_a) / sum(...)
        # Apply softmax for normalization (as in equation 1)
        exp_f_a = torch.exp(f_a)  # [B*T, N, N]
        numerator = f_p * exp_f_a  # [B*T, N, N]
        
        # Normalize: sum over j for each i
        denominator = numerator.sum(dim=2, keepdim=True) + 1e-8  # [B*T, N, 1]
        adjacency_matrix = numerator / denominator  # [B*T, N, N]
        
        # Apply GCN: Z^(l+1) = σ(A * Z^(l) * W^(l))
        # adjacency_matrix: [B*T, N, N], appearance_features: [B*T, N, D]
        gcn_output = torch.matmul(adjacency_matrix, appearance_features)  # [B*T, N, D]
        gcn_output = self.fc_gcn(gcn_output)  # [B*T, N, D]
        gcn_output = F.relu(gcn_output)  # ReLU activation
        
        return gcn_output, adjacency_matrix


class Motion_GCN_Module(nn.Module):
    """
    Motion GCN Module as described in the paper
    Adjacency matrix: b_sim,t^ij = 1/||u_t^i - u_t^j||_2
    where u_t^i = [c_x, c_y] is the 2D centroid of bounding box
    """
    def __init__(self, cfg):
        super(Motion_GCN_Module, self).__init__()
        
        self.cfg = cfg
        NFG = cfg.num_features_gcn
        D = NFG  # Dimension of motion features
        
        # GCN weight matrix W^(l)
        self.fc_gcn = nn.Linear(D, D, bias=False)
        
    def forward(self, motion_features, boxes_in_flat, presence_mask=None):
        """
        Args:
            motion_features: [B*T, N, D] motion features
            boxes_in_flat: [B*T*N, 4] bounding boxes
            presence_mask: [B*T, N] indicator for present tracklets (optional)
        Returns:
            output_features: [B*T, N, D] GCN output features
            adjacency_matrix: [B*T, N, N] motion adjacency matrix
        """
        B_T, N, D = motion_features.shape
        
        # Compute 2D centroids u_t^i = [c_x, c_y] from bounding boxes
        boxes_reshaped = boxes_in_flat.reshape(B_T, N, 4)  # [B*T, N, 4]
        centroids = torch.zeros(B_T, N, 2, device=boxes_reshaped.device, dtype=boxes_reshaped.dtype)
        centroids[:, :, 0] = (boxes_reshaped[:, :, 0] + boxes_reshaped[:, :, 2]) / 2.0  # c_x
        centroids[:, :, 1] = (boxes_reshaped[:, :, 1] + boxes_reshaped[:, :, 3]) / 2.0  # c_y
        
        # Compute pairwise distances ||u_t^i - u_t^j||_2
        # centroids: [B*T, N, 2]
        pairwise_distances = calc_pairwise_distance_3d(centroids, centroids)  # [B*T, N, N]
        
        # Compute presence mask if not provided
        if presence_mask is None:
            presence_mask = (boxes_reshaped.abs().sum(dim=2) > 1e-6).float()  # [B*T, N]
        
        # Compute b_sim,t^ij according to equation 5
        adjacency_matrix = torch.zeros_like(pairwise_distances)  # [B*T, N, N]
        
        # Case 1: ||u_t^i - u_t^j||_2 != 0: b_sim,t^ij = 1 / distance
        mask_nonzero = pairwise_distances > 1e-6
        adjacency_matrix[mask_nonzero] = 1.0 / pairwise_distances[mask_nonzero]
        
        # Case 2: u_t^i OR u_t^j missing: b_sim,t^ij = 1
        mask_missing = (presence_mask.unsqueeze(2) < 1e-6) | (presence_mask.unsqueeze(1) < 1e-6)
        adjacency_matrix[mask_missing] = 1.0
        
        # Case 3: Otherwise (both missing or invalid): b_sim,t^ij = 0
        # Already handled by zero initialization
        
        # Normalize adjacency matrix (optional, but helps stability)
        # For motion graph, we can apply row-wise normalization
        row_sum = adjacency_matrix.sum(dim=2, keepdim=True) + 1e-8
        adjacency_matrix = adjacency_matrix / row_sum
        
        # Apply GCN: Z^(l+1) = σ(A * Z^(l) * W^(l))
        gcn_output = torch.matmul(adjacency_matrix, motion_features)  # [B*T, N, D]
        gcn_output = self.fc_gcn(gcn_output)  # [B*T, N, D]
        gcn_output = F.relu(gcn_output)  # ReLU activation
        
        return gcn_output, adjacency_matrix


class DualGCN_Model(nn.Module):
    """
    Dual GCN Model as described in the paper
    Combines Appearance GCN and Motion GCN with late fusion
    """
    def __init__(self, cfg):
        super(DualGCN_Model, self).__init__()
        self.cfg = cfg
        
        T, N = cfg.num_frames, cfg.num_boxes
        D = cfg.emb_features
        K = cfg.crop_size[0]
        NFB = cfg.num_features_boxes
        NFG = cfg.num_features_gcn
        num_gcn_layers = cfg.gcn_layers if hasattr(cfg, 'gcn_layers') else 1
        
        # Backbone for appearance features
        if cfg.backbone == 'inv3':
            self.backbone = MyInception_v3(transform_input=False, pretrained=True)
        elif cfg.backbone == 'vgg16':
            self.backbone = MyVGG16(pretrained=True)
        elif cfg.backbone == 'vgg19':
            self.backbone = MyVGG19(pretrained=True)
        else:
            assert False
        
        if not cfg.train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.roi_align = RoIAlign(*cfg.crop_size)
        
        # Appearance feature embedding
        self.fc_emb_appearance = nn.Linear(K*K*D, NFB)
        self.nl_emb_appearance = nn.LayerNorm([NFB])
        
        # Motion feature embedding (project 2D centroids to feature space)
        # Motion features are extracted from 2D centroids
        self.fc_emb_motion = nn.Linear(2, NFB)  # 2D centroid -> feature dimension
        
        # Multiple GCN layers
        self.appearance_gcn_layers = nn.ModuleList([
            Appearance_GCN_Module(cfg) for _ in range(num_gcn_layers)
        ])
        
        self.motion_gcn_layers = nn.ModuleList([
            Motion_GCN_Module(cfg) for _ in range(num_gcn_layers)
        ])
        
        self.dropout = nn.Dropout(p=cfg.train_dropout_prob)
        
        # Final classification layers (multi-label)
        # For multi-label classification, we output probabilities for each class
        self.fc_activities = nn.Linear(NFG * 2, cfg.num_activities)  # *2 for fusion of appearance + motion
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def loadmodel(self, filepath):
        """Load backbone weights from stage1 model"""
        state = torch.load(filepath)
        self.backbone.load_state_dict(state['backbone_state_dict'])
        if 'fc_emb_state_dict' in state:
            self.fc_emb_appearance.load_state_dict(state['fc_emb_state_dict'])
        print('Load model states from:', filepath)
    
    def forward(self, batch_data):
        """
        Forward pass for dual GCN model
        Args:
            batch_data: (images_in, boxes_in) or (images_in, boxes_in, bboxes_num_in)
        Returns:
            activities_scores: [B, num_activities] multi-label scores
        """
        if len(batch_data) == 3:
            images_in, boxes_in, bboxes_num_in = batch_data
        else:
            images_in, boxes_in = batch_data
            bboxes_num_in = None
        
        B = images_in.shape[0]
        T = images_in.shape[1]
        H, W = self.cfg.image_size
        OH, OW = self.cfg.out_size
        N = self.cfg.num_boxes
        NFB = self.cfg.num_features_boxes
        NFG = self.cfg.num_features_gcn
        D = self.cfg.emb_features
        K = self.cfg.crop_size[0]
        
        # Reshape inputs
        images_in_flat = images_in.reshape(B*T, 3, H, W)  # [B*T, 3, H, W]
        boxes_in_flat = boxes_in.reshape(B*T*N, 4)  # [B*T*N, 4]
        
        boxes_idx = [i * torch.ones(N, dtype=torch.int) for i in range(B*T)]
        boxes_idx = torch.stack(boxes_idx).to(device=boxes_in.device)  # [B*T, N]
        boxes_idx_flat = boxes_idx.reshape(B*T*N)  # [B*T*N]
        
        # ========== Appearance Feature Extraction ==========
        # Preprocess images
        images_in_flat = prep_images(images_in_flat)
        outputs = self.backbone(images_in_flat)
        
        # Build multiscale features
        features_multiscale = []
        for features in outputs:
            if features.shape[2:4] != torch.Size([OH, OW]):
                features = F.interpolate(features, size=(OH, OW), mode='bilinear', align_corners=True)
            features_multiscale.append(features)
        features_multiscale = torch.cat(features_multiscale, dim=1)  # [B*T, D, OH, OW]
        
        # RoI Align for appearance features
        boxes_in_flat.requires_grad = False
        boxes_idx_flat.requires_grad = False
        appearance_roi_features = self.roi_align(features_multiscale, boxes_in_flat, boxes_idx_flat)  # [B*T*N, D, K, K]
        appearance_roi_features = appearance_roi_features.reshape(B*T, N, -1)  # [B*T, N, D*K*K]
        
        # Embed appearance features
        appearance_features = self.fc_emb_appearance(appearance_roi_features)  # [B*T, N, NFB]
        appearance_features = self.nl_emb_appearance(appearance_features)
        appearance_features = F.relu(appearance_features)
        
        # Project to GCN feature dimension if needed
        if NFB != NFG:
            if not hasattr(self, 'fc_appearance_to_gcn'):
                self.fc_appearance_to_gcn = nn.Linear(NFB, NFG).to(appearance_features.device)
                nn.init.kaiming_normal_(self.fc_appearance_to_gcn.weight)
            appearance_features = self.fc_appearance_to_gcn(appearance_features)  # [B*T, N, NFG]
        
        # ========== Motion Feature Extraction ==========
        # Extract 2D centroids from bounding boxes
        boxes_reshaped = boxes_in_flat.reshape(B*T, N, 4)  # [B*T, N, 4]
        centroids = torch.zeros(B*T, N, 2, device=boxes_reshaped.device, dtype=boxes_reshaped.dtype)
        centroids[:, :, 0] = (boxes_reshaped[:, :, 0] + boxes_reshaped[:, :, 2]) / 2.0  # c_x
        centroids[:, :, 1] = (boxes_reshaped[:, :, 1] + boxes_reshaped[:, :, 3]) / 2.0  # c_y
        
        # Embed motion features (2D centroids -> feature space)
        motion_features = self.fc_emb_motion(centroids)  # [B*T, N, NFB]
        motion_features = F.relu(motion_features)
        
        # Project to GCN feature dimension if needed
        if NFB != NFG:
            if not hasattr(self, 'fc_motion_to_gcn'):
                self.fc_motion_to_gcn = nn.Linear(NFB, NFG).to(motion_features.device)
                nn.init.kaiming_normal_(self.fc_motion_to_gcn.weight)
            motion_features = self.fc_motion_to_gcn(motion_features)  # [B*T, N, NFG]
        
        # ========== Appearance GCN ==========
        # Compute presence mask
        presence_mask = (boxes_reshaped.abs().sum(dim=2) > 1e-6).float()  # [B*T, N]
        
        for gcn_layer in self.appearance_gcn_layers:
            appearance_features, _ = gcn_layer(appearance_features, boxes_in_flat, presence_mask)
        
        # ========== Motion GCN ==========
        for gcn_layer in self.motion_gcn_layers:
            motion_features, _ = gcn_layer(motion_features, boxes_in_flat, presence_mask)
        
        # ========== Late Fusion ==========
        # Pool features across nodes (max pooling over N agents)
        appearance_pooled, _ = torch.max(appearance_features, dim=1)  # [B*T, NFG]
        motion_pooled, _ = torch.max(motion_features, dim=1)  # [B*T, NFG]
        
        # Concatenate appearance and motion features
        fused_features = torch.cat([appearance_pooled, motion_pooled], dim=1)  # [B*T, NFG*2]
        fused_features = self.dropout(fused_features)
        
        # Temporal pooling (average over time)
        fused_features = fused_features.reshape(B, T, -1)  # [B, T, NFG*2]
        fused_features = torch.mean(fused_features, dim=1)  # [B, NFG*2]
        
        # ========== Multi-label Classification ==========
        # For multi-label classification, output logits (sigmoid will be applied in loss)
        activities_scores = self.fc_activities(fused_features)  # [B, num_activities]
        
        return activities_scores

