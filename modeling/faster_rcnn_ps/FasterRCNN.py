import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class FasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone,
                 num_classes=2, num_pids=5532, num_cq_size=5000,
                 # transform parameters
                 min_size=900, max_size=1500,
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
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 feat_head=None,
                 reid_head=None, reid_loss=None):
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator= AnchorGenerator(anchor_sizes,aspect_ratios)
        rpn_head=RPNHead(backbone.out_channels,rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn=RegionProposalNetwork(rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)


        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat2rpn'],
                output_size=[7,7],
                sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 2048
            box_head = TwoMLPHead(
                backbone.out_channels*resolution**2 ,
                representation_size
            )
        if box_predictor is None:
            representation_size=2048
            box_predictor=FastRCNNPredictor(representation_size,num_classes)
        roi_heads=RoIHeads(            # box
                                box_roi_pool, box_head, box_predictor,
                                box_fg_iou_thresh, box_bg_iou_thresh,
                                box_batch_size_per_image, box_positive_fraction,
                                bbox_reg_weights,
                                box_score_thresh, box_nms_thresh, box_detections_per_img)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        super(FasterRCNN, self).__init__(backbone,rpn,roi_heads,transform)
class TwoMLPHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(TwoMLPHead,self).__init__()
        self.fc1=nn.Linear(in_channels,out_channels)
        self.fc2=nn.Linear(out_channels,out_channels)
    def forward(self,x):
        x=x.flatten(start_dim=1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x

class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

