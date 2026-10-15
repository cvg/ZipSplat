# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DA3's camera decoder: per-view camera token -> pose encoding [t(3), quat_xyzw(4), fov(2)].

Verbatim port of the DA3 cam_dec. The DA3 checkpoints ship its pretrained `cam_dec.*` weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from splatfactory.geometry.cameras import Camera
from splatfactory.geometry.poses import Pose
from splatfactory.utils.conversions import quaternion_to_matrix


class CameraDec(nn.Module):
    def __init__(self, dim_in=1536):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat):
        B, N = feat.shape[:2]
        feat = self.backbone(feat.reshape(B * N, -1))
        out_t = self.fc_t(feat.float()).reshape(B, N, 3)
        out_qvec = self.fc_qvec(feat.float()).reshape(B, N, 4)
        out_fov = self.fc_fov(feat.float()).reshape(B, N, 2)
        return torch.cat([out_t, out_qvec, out_fov], dim=-1)  # [B, N, 9]


def decode_pose_encoding(pose_enc, image_hw):
    """Pose encoding [..., 9] (t | quat_xyzw | fov_h, fov_w) -> (c2w Pose, Camera)."""
    t, quat, fov = pose_enc[..., :3], pose_enc[..., 3:7], pose_enc[..., 7:]
    quat = F.normalize(quat, dim=-1)  # quaternion_to_matrix expects xyzw (= enc order)
    pose = Pose.from_Rt(quaternion_to_matrix(quat), t)
    H, W = image_hw
    fov = fov.clamp(min=0.02, max=3.12)  # guard: ReLU fov head can hit 0 -> inf focal
    camera = Camera.from_fov(fov[..., 1], fov[..., 0], w=W, h=H)
    return pose, camera
