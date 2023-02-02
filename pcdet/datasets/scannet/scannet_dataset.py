import copy
import pickle
import warnings
import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from ..augmentor.data_augmentor import DataAugmentor

class ScannetDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, filter_empty_gt=True):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.repeat = self.dataset_cfg.REPEAT[self.mode]
        self.cat2id = {name: i for i, name in enumerate(self.class_names)}
        self.sample_id_list = []
        self.scannet_infos = []
        self.filter_empty_gt = filter_empty_gt
        self.include_scannet_data(self.mode)
        self.reload_data_augmentor()

    def reload_data_augmentor(self):
        self.data_augmentor_train = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR_TRAIN, self.class_names, logger=self.logger)
        self.data_augmentor_test = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR_TEST, self.class_names, logger=self.logger)

    def include_scannet_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading SCANNET dataset')
        scannet_infos = []
        sample_id_list = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                scannet_infos.extend(infos)
                sample_id_list.extend([info['point_cloud']['lidar_idx'] for info in infos])
        
        for _ in range(self.repeat):
            self.sample_id_list.extend(sample_id_list)
            self.scannet_infos.extend(scannet_infos)

        if self.logger is not None:
            self.logger.info('Total samples for SCANNET dataset: %d' % (len(scannet_infos)))


    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 6)

    def get_instance_mask(self, idx):
        lidar_file = self.root_path / 'instance_mask' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.long)

    def get_semantic_mask(self, idx):
        lidar_file = self.root_path / 'semantic_mask' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.long)

    def get_axis_align_matrix(self, info):
        if 'axis_align_matrix' in info['annos'].keys():
            return copy.deepcopy(np.array(info['annos']['axis_align_matrix']).astype(np.float32))
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)


    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'labels_3d': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'scores_3d': np.zeros(num_samples), 'boxes_3d': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels]
            pred_dict['labels_3d'] = pred_labels
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['scores_3d'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)
    
            if output_path is not None:
                raise NotImplementedError

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .scannet_object_eval_python import eval as scannet_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.scannet_infos]
        metric_thresholds = [0.25, 0.5]
        label2cat = {i: cat_id for i, cat_id in enumerate(class_names)}
        ret_dict = scannet_eval.indoor_eval(eval_gt_annos, eval_det_annos, metric_thresholds, label2cat)

        return ret_dict, ret_dict

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor_train.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )
        else:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor_test.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )


        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict.pop('gt_names', None)
        return data_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.scannet_infos) * self.total_epochs

        return len(self.scannet_infos)
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        index = np.random.randint(len(self))
        return index

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.scannet_infos)
        info = copy.deepcopy(self.scannet_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {'frame_id': sample_idx}

        if info['annos']['gt_num'] != 0:
            annos = info['annos']
            # align gt
            loc, dims, rots = annos['location'], annos['dimensions'], np.zeros((len(annos['location'])))
            gt_names = annos['name']
            gt_boxes_depth = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_depth
            })
        else:
            input_dict.update({
                'gt_names': np.array([]),
                'gt_boxes': np.zeros((0, 7), dtype=np.float32),
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        if "instance_mask" in get_item_list:
            instance_mask = self.get_instance_mask(sample_idx)
            input_dict['instance_mask'] = instance_mask

        if "semantic_mask" in get_item_list:
            semantic_mask = self.get_semantic_mask(sample_idx)
            input_dict['semantic_mask'] = semantic_mask

        axis_align_matrix = self.get_axis_align_matrix(info)
        input_dict['axis_align_matrix'] = axis_align_matrix

        if not self.training:
            data_dict = self.prepare_data(data_dict=input_dict)
            return data_dict
        
        data_dict = self.prepare_data(data_dict=input_dict)
        if len(data_dict['gt_boxes']) == 0 and self.filter_empty_gt:
            index = self._rand_another(index)
            data_dict = self.__getitem__(index)
        return data_dict

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    raise NotImplementedError

if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
