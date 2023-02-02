import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

def parse_params(param_edict):
    """Convert upper easydict param to lower normal dict, only support depth=1 now

    Args:
        param_edict: easydict contains parameters
    """
    out = dict()
    for k, v in param_edict.items():
        if k.lower() == 'name':
            continue
        out.update({k.lower(): v})
    return out

def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

def bias_init_with_prob(prior_prob):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init

class CAGroupResidualCoder(object):
    def __init__(self, code_size=6, encode_angle_by_sincos=False, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1
    
    def encode_torch(self, boxes, anchors):
        """
        Args:
            boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            anchors: (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]

        Returns:

        """
        anchors[:, 3:6] = torch.clamp_min(anchors[:, 3:6], min=1e-5)
        boxes[:, 3:6] = torch.clamp_min(boxes[:, 3:6], min=1e-5)

        if self.code_size > 6:
            xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
            xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(boxes, 1, dim=-1)

            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
            if self.encode_angle_by_sincos:
                # rt_cos = torch.cos(rg) - torch.cos(ra)
                # rt_sin = torch.sin(rg) - torch.sin(ra)
                rt_cos = torch.cos(rg) # directly encode delta theta
                rt_sin = torch.sin(rg)
                rts = [rt_cos, rt_sin]
            else:
                rts = [rg - ra]

            cts = [g - a for g, a in zip(cgs, cas)]
            return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)
        else:
            xa, ya, za, dxa, dya, dza = torch.split(anchors, 1, dim=-1)
            xg, yg, zg, dxg, dyg, dzg = torch.split(boxes, 1, dim=-1)

            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)

            return torch.cat([xt, yt, zt, dxt, dyt, dzt], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        if self.code_size > 6:
            xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
            if not self.encode_angle_by_sincos:
                xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
            else:
                xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza

            if self.encode_angle_by_sincos:
                # rg_cos = cost + torch.cos(ra)
                # rg_sin = sint + torch.sin(ra)
                rg_cos = cost # directly decode delta theta
                rg_sin = sint
                rg = torch.atan2(rg_sin, rg_cos)
                rg = rg + ra
            else:
                rg = rt + ra

            cgs = [t + a for t, a in zip(cts, cas)]
            return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)
        else:
            xa, ya, za, dxa, dya, dza = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, dxt, dyt, dzt = torch.split(box_encodings, 1, dim=-1)

            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa # xt: (xg - xa)/diagonal
            yg = yt * diagonal + ya
            zg = zt * dza + za # zt: (zg - za)/dza

            dxg = torch.exp(dxt) * dxa # dxt : log(dxg/dxa)
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza

            return torch.cat([xg, yg, zg, dxg, dyg, dzg], dim=-1)