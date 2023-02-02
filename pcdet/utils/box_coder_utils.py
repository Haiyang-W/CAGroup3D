import numpy as np
import torch


class ResidualCoder(object):
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
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
            rt_cos = torch.cos(rg) - torch.cos(ra)
            rt_sin = torch.sin(rg) - torch.sin(ra)
            rts = [rt_cos, rt_sin]
        else:
            rts = [rg - ra]

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, *rts, *cts], dim=-1)

    def decode_torch(self, box_encodings, anchors):
        """
        Args:
            box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or *[cos, sin], ...]
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
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
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PreviousResidualRoIDecoder(object):
    def __init__(self, code_size=7, **kwargs):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        Args:
            box_encodings:  (B, N, 7 + ?) x, y, z, w, l, h, r, custom values
            anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(lt) * dxa
        dyg = torch.exp(wt) * dya
        dzg = torch.exp(ht) * dza
        rg = ra - rt

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


class PointResidualCoder(object):
    def __init__(self, code_size=8, use_mean_size=True, **kwargs):
        super().__init__()
        self.code_size = code_size
        self.use_mean_size = use_mean_size
        if self.use_mean_size:
            self.mean_size = torch.from_numpy(np.array(kwargs['mean_size'])).cuda().float()
            assert self.mean_size.min() > 0

    def encode_torch(self, gt_boxes, points, gt_classes=None):
        """
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            points: (N, 3) [x, y, z]
            gt_classes: (N) [1, num_classes]
        Returns:
            box_coding: (N, 8 + C)
        """
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)

        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert gt_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[gt_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xt = (xg - xa) / diagonal
            yt = (yg - ya) / diagonal
            zt = (zg - za) / dza
            dxt = torch.log(dxg / dxa)
            dyt = torch.log(dyg / dya)
            dzt = torch.log(dzg / dza)
        else:
            xt = (xg - xa)
            yt = (yg - ya)
            zt = (zg - za)
            dxt = torch.log(dxg)
            dyt = torch.log(dyg)
            dzt = torch.log(dzg)

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, dxt, dyt, dzt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

    def decode_torch(self, box_encodings, points, pred_classes=None):
        """
        Args:
            box_encodings: (N, 8 + C) [x, y, z, dx, dy, dz, cos, sin, ...]
            points: [x, y, z]
            pred_classes: (N) [1, num_classes]
        Returns:

        """
        xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)
        xa, ya, za = torch.split(points, 1, dim=-1)

        if self.use_mean_size:
            assert pred_classes.max() <= self.mean_size.shape[0]
            point_anchor_size = self.mean_size[pred_classes - 1]
            dxa, dya, dza = torch.split(point_anchor_size, 1, dim=-1)
            diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
            xg = xt * diagonal + xa
            yg = yt * diagonal + ya
            zg = zt * dza + za

            dxg = torch.exp(dxt) * dxa
            dyg = torch.exp(dyt) * dya
            dzg = torch.exp(dzt) * dza
        else:
            xg = xt + xa
            yg = yt + ya
            zg = zt + za
            dxg, dyg, dzg = torch.split(torch.exp(box_encodings[..., 3:6]), 1, dim=-1)

        rg = torch.atan2(sint, cost)

        cgs = [t for t in cts]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)

class RBGBBoxCoder(object):
    def __init__(self, ray_num, num_dir_bins, num_sizes, with_rot=True):
        super(RBGBBoxCoder, self).__init__()
        self.ray_num = ray_num
        self.num_dir_bins = num_dir_bins
        self.num_sizes = num_sizes
        self.with_rot = with_rot

    def encode(self, gt_bboxes_3d, gt_labels_3d, ret_dir_target=False):
        center_target = gt_bboxes_3d[:, :3]

        size_class_target = gt_labels_3d
        size_target = gt_bboxes_3d[:, 3:6]

        size_half_target = gt_bboxes_3d[:, 3:6].clone() / 2

        scale_class_targets = gt_labels_3d
        scale_targets = torch.linalg.norm(gt_bboxes_3d[:, 3:6], dim=1)
        scale_targets = scale_targets.unsqueeze(-1)

        # generate dir target
        box_num = gt_labels_3d.shape[0]

        if self.with_rot:
            (dir_class_target,
             dir_res_target) = self.angle2class(gt_bboxes_3d[:, 6])
            dir_target = gt_bboxes_3d[:, 6]
        else:
            dir_class_target = gt_labels_3d.new_zeros(box_num)
            dir_res_target = gt_bboxes_3d.new_zeros(box_num)
            dir_target = gt_bboxes_3d.new_zeros(box_num)

        if ret_dir_target:
            return (center_target, size_half_target, dir_class_target, dir_res_target, dir_target, 
                        size_class_target, size_target, scale_class_targets, scale_targets)
        else:
            return (center_target, size_half_target, dir_class_target, dir_res_target, 
                        size_class_target, size_target, scale_class_targets, scale_targets)

    def scale_pred(self, reg_preds, ref_points):
        results = {}
        start, end = 0, 0

        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode scale class
        end += 1
        scale_res_norm = reg_preds_trans[..., start:end]
        batch_size, num_proposal = reg_preds_trans.shape[:2]
        scale_res_norm = scale_res_norm.view(
            [batch_size, num_proposal, 1, 1])
        results['scale_res_norm'] = scale_res_norm.contiguous()

        results['scale_size_res'] = torch.exp(results['scale_res_norm'])

        results['ref_points'] = ref_points  # (batch_size, num_proposal, 3)

        return results

    def split_pred(self, cls_preds, reg_preds, base_xyz):
        results = {}
        start, end = 0, 0

        cls_preds_trans = cls_preds.transpose(2, 1)
        reg_preds_trans = reg_preds.transpose(2, 1)

        # decode center
        end += 3
        # (batch_size, num_proposal, 3)
        results['center'] = base_xyz + \
                            reg_preds_trans[..., start:end].contiguous()
        start = end

        # decode direction
        end += self.num_dir_bins
        results['dir_class'] = reg_preds_trans[..., start:end].contiguous()
        start = end

        end += self.num_dir_bins
        dir_res_norm = reg_preds_trans[..., start:end].contiguous()
        start = end

        results['dir_res_norm'] = dir_res_norm
        results['dir_res'] = dir_res_norm * (np.pi / self.num_dir_bins)

        # decode size
        end += 3
        size_res_norm = reg_preds_trans[..., start:end]
        batch_size, num_proposal = reg_preds_trans.shape[:2]
        size_res_norm = size_res_norm.view(
            [batch_size, num_proposal, 1, 3])

        results['size_res_norm'] = size_res_norm.contiguous()
        results['size_res'] = torch.exp(size_res_norm)
        start = end

        # decode objectness score
        start = 0
        end = 2
        results['obj_scores'] = cls_preds_trans[..., start:end].contiguous()
        start = end

        # decode semantic score
        results['sem_scores'] = cls_preds_trans[..., start:].contiguous()

        return results

    def decode_bbox(self, bbox_out):
        center = bbox_out['center']
        batch_size, num_proposal = center.shape[:2]

        # decode heading angle
        if self.with_rot:
            dir_class = torch.argmax(bbox_out['dir_class'], -1)
            dir_res = torch.gather(bbox_out['dir_res'], 2,
                                   dir_class.unsqueeze(-1))
            dir_res.squeeze_(2)
            dir_angle = self.class2angle(dir_class, dir_res).reshape(
                batch_size, num_proposal, 1)
        else:
            dir_angle = center.new_zeros(batch_size, num_proposal, 1)

        bbox_size = bbox_out['size_res'].squeeze(2)
        bbox3d = torch.cat([center, bbox_size, dir_angle], dim=-1)
        return bbox3d

    def decode_scale(self, scale_size_res_pred):
        batch_size = scale_size_res_pred.shape[0]
        bbox_scale = scale_size_res_pred.view(batch_size, -1)

        return bbox_scale

    def decode_corners(self, center, size_res):
        size_full = size_res
        size_full = torch.clamp(size_full, 0)
        half_size_full = size_full / 2
        corner1 = center - half_size_full
        corner2 = center + half_size_full
        corners = torch.cat([corner1, corner2], dim=-1)
        return corners

    def angle2class(self, angle):
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        angle_cls = shifted_angle // angle_per_class
        angle_res = shifted_angle - (
                angle_cls * angle_per_class + angle_per_class / 2)
        return angle_cls.long(), angle_res

    def class2angle(self, angle_cls, angle_res, limit_period=True):
        angle_per_class = 2 * np.pi / float(self.num_dir_bins)
        angle_center = angle_cls.float() * angle_per_class
        angle = angle_center + angle_res
        if limit_period:
            angle[angle > np.pi] -= 2 * np.pi
        return angle

