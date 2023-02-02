from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST
        
        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)
    
    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d
    
    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)

        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_rotation_mmdet3d(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation_mmdet3d, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]

        gt_boxes, points = augmentor_utils.global_rotation_mmdet3d(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )
        
        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict
    
    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        noise_translate_std = config['NOISE_TRANSLATE_STD']
        if noise_translate_std == 0:
            return data_dict
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_translation_along_%s' % cur_axis)(
                gt_boxes, points, noise_translate_std,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_local_translation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_translation, config=config)
        offset_range = config['LOCAL_TRANSLATION_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y', 'z']
            gt_boxes, points = getattr(augmentor_utils, 'random_local_translation_along_%s' % cur_axis)(
                gt_boxes, points, offset_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_rotation(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_rotation, config=config)
        rot_range = config['LOCAL_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.local_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_scaling(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_scaling, config=config)
        gt_boxes, points = augmentor_utils.local_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['LOCAL_SCALE_RANGE']
        )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_world_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_world_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'global_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_frustum_dropout(self, data_dict=None, config=None):
        """
        Please check the correctness of it before using.
        """
        if data_dict is None:
            return partial(self.random_local_frustum_dropout, config=config)
        
        intensity_range = config['INTENSITY_RANGE']
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for direction in config['DIRECTION']:
            assert direction in ['top', 'bottom', 'left', 'right']
            gt_boxes, points = getattr(augmentor_utils, 'local_frustum_dropout_%s' % direction)(
                gt_boxes, points, intensity_range,
            )
        
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict
    
    def random_local_pyramid_aug(self, data_dict=None, config=None):
        """
        Refer to the paper: 
            SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud
        """
        if data_dict is None:
            return partial(self.random_local_pyramid_aug, config=config)
        
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_dropout(gt_boxes, points, config['DROP_PROB'])
        gt_boxes, points, pyramids = augmentor_utils.local_pyramid_sparsify(gt_boxes, points,
                                                                            config['SPARSIFY_PROB'],
                                                                            config['SPARSIFY_MAX_NUM'],
                                                                            pyramids)
        gt_boxes, points = augmentor_utils.local_pyramid_swap(gt_boxes, points,
                                                                 config['SWAP_PROB'],
                                                                 config['SWAP_MAX_NUM'],
                                                                 pyramids)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def global_alignment(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.global_alignment, config=config)

        assert 'axis_align_matrix' in data_dict.keys(), \
            'axis_align_matrix is not provided in GlobalAlignment'
        axis_align_matrix = data_dict['axis_align_matrix']
        assert axis_align_matrix.shape == (4, 4), \
            f'invalid shape {axis_align_matrix.shape} for axis_align_matrix'
        points = data_dict['points']
        points = augmentor_utils.global_alignment(points, axis_align_matrix, config['rotation_axis'])

        data_dict['points'] = points

        return data_dict

    def point_seg_class_mapping(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.point_seg_class_mapping, config=config)

        assert 'semantic_mask' in data_dict.keys(), \
            'semantic_mask is not provided in PointSegClassMapping'
        semantic_mask = data_dict['semantic_mask']
        semantic_mask = augmentor_utils.point_seg_class_mapping(semantic_mask,
                                                                config['valid_cat_ids'],
                                                                config['max_cat_id'])
        data_dict['semantic_mask'] = semantic_mask
        return data_dict

    def indoor_point_sample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.indoor_point_sample, config=config)

        points = data_dict['points']
        points, choices = augmentor_utils.points_random_sampling(points, config['num_points'], return_choices=True)
        data_dict['points'] = points

        instance_mask = data_dict.get('instance_mask', None)
        semantic_mask = data_dict.get('semantic_mask', None)

        if instance_mask is not None:
            instance_mask = instance_mask[choices]
            data_dict['instance_mask'] = instance_mask

        if semantic_mask is not None:
            semantic_mask = semantic_mask[choices]
            data_dict['semantic_mask'] = semantic_mask

        return data_dict
    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )

        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]

            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
            
            data_dict.pop('gt_boxes_mask')
        return data_dict
