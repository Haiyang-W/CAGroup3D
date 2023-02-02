from .detector3d_template import Detector3DTemplate
import torch


class RBGNet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # normalize point features
        batch_dict['points'][:, -3:] = batch_dict['points'][:, -3:] / 255.
        for cur_module in self.module_list:
            results = cur_module(batch_dict)
            batch_dict.update(results)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            recall_dict = self.generate_recall_record(
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            record_dict = {
                'pred_boxes':  batch_dict['batch_box_preds'][index],
                'pred_scores': batch_dict['batch_score_preds'][index],
                'pred_labels': batch_dict['batch_cls_preds'][index]
            }
            pred_dicts.append(record_dict)
        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        return recall_dict

    def get_training_loss(self, batch_dict):
        batch_size = batch_dict['batch_size']
        if 'instance_mask' in batch_dict.keys():
            pts_instance_mask = batch_dict['instance_mask'].view(batch_size, -1) 
        else:
            pts_instance_mask = None
        if 'semantic_mask' in batch_dict.keys():
            pts_semantic_mask = batch_dict['semantic_mask'].view(batch_size, -1) 
        else:
            pts_semantic_mask = None

        gt_bboxes_3d = []
        gt_labels_3d = []
        device = batch_dict['points'].device
        for b in range(len(batch_dict['gt_boxes'])):
            gt_bboxes_b = []
            gt_labels_b = []
            for _item in batch_dict['gt_boxes'][b]:
                if not (_item == 0.).all(): 
                    gt_bboxes_b.append(_item[:7])  
                    gt_labels_b.append(_item[7:8]) 
            if len(gt_bboxes_b) == 0:
                gt_bboxes_b = torch.zeros((0, 7), dtype=torch.float32).to(device)
                gt_labels_b = torch.zeros((0,), dtype=torch.int).to(device)
            else:
                gt_bboxes_b = torch.stack(gt_bboxes_b)
                gt_labels_b = torch.cat(gt_labels_b).int()
            gt_bboxes_3d.append(gt_bboxes_b)
            gt_labels_3d.append(gt_labels_b)
        img_metas = [None for _ in range(batch_size)]

        losses_inputs = (batch_dict['points_cat'], gt_bboxes_3d, gt_labels_3d,
                         pts_semantic_mask, pts_instance_mask, img_metas)
        disp_dict = {}
        loss_all, tb_dict = self.point_head.loss(batch_dict, *losses_inputs)
        for k in tb_dict.keys():
            disp_dict[k] = tb_dict[k].item()
        tb_dict = {
            'loss_all': loss_all.item(),
            **tb_dict
        }

        loss = loss_all
        return loss, tb_dict, disp_dict
