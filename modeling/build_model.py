from .backbone import resnet
import torch.nn as nn
import torch
from collections import OrderedDict
from .faster_rcnn_ps import FasterRCNN
import modeling.faster_rcnn_ps as ps

def build(cfg):
    ori_backbone = resnet.__dict__[cfg.MODEL.BACKBONE](pretrained=True)
    backbone, feat_head = get_backbone(ori_backbone)


    model = ps.__dict__[cfg.MODEL.METHOD](backbone,
                                           num_classes=2, num_pids=5532, num_cq_size=5000,
                                           # transform parameters
                                           min_size=cfg.INPUT.MIN_MAX_SIZE[0], max_size=cfg.INPUT.MIN_MAX_SIZE[1],
                                           image_mean=None, image_std=None,
                                           # Anchor settings:
                                           anchor_scales=None, anchor_ratios=None,
                                           # RPN parameters
                                           rpn_anchor_generator=None, rpn_head=None,
                                           rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                                           rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                                           rpn_nms_thresh=0.7,
                                           rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                                           rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                                           # Box parameters
                                           rcnn_bbox_bn=True,
                                           box_roi_pool=None, box_head=None, box_predictor=None,
                                           box_score_thresh=0.5, box_nms_thresh=0.4, box_detections_per_img=300,
                                           box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                                           box_batch_size_per_image=128, box_positive_fraction=0.5,
                                           bbox_reg_weights=None,
                                           # ReID parameters
                                           feat_head=feat_head,
                                           reid_head=None, reid_loss=None)
    # r1=backbone(a)
    # r2=feat_head(r1['feat2rpn'])
    # r3=ori_backbone(a)
    # model.eval()
    # model(a)
    return model
    pass


class FasterRCNN_Backbone(nn.Module):
    def __init__(self, ori_backbone):
        super(FasterRCNN_Backbone, self).__init__()
        self.backbone = nn.Sequential(*list(ori_backbone.children())[:-3])
        self.out_channels = 1024

    def forward(self, x):
        feat2rpn = self.backbone(x)
        return OrderedDict([['feat2rpn', feat2rpn]])


class FasterRCNN_FeatHead(nn.Module):
    def __init__(self, ori_backbone):
        super(FasterRCNN_FeatHead, self).__init__()
        self.feat_head = nn.Sequential(*list(ori_backbone.children())[-3])
        self.out_channels = 2048
        self.stride=2

    def forward(self, x):
        feat = self.feat_head(x)
        return feat


def get_backbone(ori_backbone):
    # freeze
    ori_backbone.conv1.weight.requires_grad = False
    ori_backbone.bn1.weight.requires_grad = False
    ori_backbone.bn1.bias.requires_grad = False
    backbone = FasterRCNN_Backbone(ori_backbone)
    feat_head = FasterRCNN_FeatHead(ori_backbone)

    return backbone, feat_head

