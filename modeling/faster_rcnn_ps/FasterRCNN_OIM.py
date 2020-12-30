import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork,concat_box_prediction_layers
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from collections import OrderedDict
from modeling.loss_func.oim_based import OIMLoss
class FasterRCNN_OIM(GeneralizedRCNN):
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
                 box_score_thresh=0.05, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 feat_head=None,
                 reid_head=None, reid_loss=None):
        if rpn_anchor_generator is None:
            anchor_sizes = ((32, 64, 128, 256, 512),)
            aspect_ratios = ((0.5, 1.0, 2.0),)
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
                output_size=[14,14],
                sampling_ratio=2)
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 2048
            box_head = GAP_BOX_HEAD(
                resolution ,
                feat_head,
                representation_size
            )
        if box_predictor is None:
            representation_size = 2048
            box_predictor = FastRCNNPredictor(representation_size, num_classes,RCNN_bbox_bn=False)
        if reid_head is None:
            reid_head = REID_HEAD(box_head.out_dims,256)
        if reid_loss is None:
            reid_loss= OIMLoss(256,num_pids,num_cq_size,0.5,30)
        roi_heads=OIM_ROI_HEAD(
                                reid_head,reid_loss,
                                # box
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
        super(FasterRCNN_OIM, self).__init__(backbone,rpn,roi_heads,transform)

    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        rcnn_features = self.roi_heads.box_head(roi_pooled_features)

        person_feat = self.roi_heads.reid_head(rcnn_features)
        return person_feat.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes,RCNN_bbox_bn=True):
        super(FastRCNNPredictor, self).__init__()
        if RCNN_bbox_bn:
            self.cls_score = nn.Sequential(
                nn.Linear(in_channels,num_classes),
                nn.BatchNorm1d(num_classes)
            )
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels,4*num_classes),
                nn.BatchNorm1d(4*num_classes)
            )
        else:
            self.cls_score = nn.Linear(in_channels, num_classes)
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas

class BOX_HEAD(nn.Module):
    def __init__(self,resolution,feat_head,representation_size):
        super(BOX_HEAD, self).__init__()
        self.feat_head=feat_head
        self.fc1=nn.Linear(feat_head.out_channels*(math.ceil(resolution/feat_head.stride))**2,representation_size)
        self.fc2=nn.Linear(representation_size,representation_size)
        self.out_dims=representation_size
    def forward(self,x):
        x=self.feat_head(x)
        x=x.flatten(start_dim=1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return x

class GAP_BOX_HEAD(nn.Module):
    def __init__(self,resolution,feat_head,representation_size):
        super(GAP_BOX_HEAD,self).__init__()
        self.feat_head=feat_head
        self.out_dims=representation_size

    def forward(self,x):
        x= self.feat_head(x)
        x= F.adaptive_max_pool2d(x,1).squeeze()
        return x
class REID_HEAD(nn.Module):
    def __init__(self, input_dims, feat_dims):
        super(REID_HEAD,self).__init__()
        self.feat_dims=feat_dims
        self.fc1= nn.Linear(input_dims, feat_dims)
        # self.fc2= nn.Linear(feat_dims, num_pids)
        # self.sm= nn.Softmax()
    def forward(self,x):
        x=F.relu(self.fc1(x))
        if len (x.shape)==1:
            x=x.unsqueeze(0)
        norms = x.norm(2, 1, keepdim=True)
        x = x / norms.expand_as(x).clamp(min=1e-12)
        return x
class OIM_ROI_HEAD(RoIHeads):
    def __init__(self,reid_head,reid_loss,*args,**kwargs):
        super(OIM_ROI_HEAD,self).__init__(*args,**kwargs)
        self.reid_head= reid_head
        self.reid_loss= reid_loss

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, \
                    'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, \
                        'target keypoints must of float type'

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        # from IPython import embed
        # embed()
        roi_pooled_features = \
            self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.box_head(roi_pooled_features)
        cls_logits,box_regression = self.box_predictor(rcnn_features)

        person_feat = self.reid_head(rcnn_features)

        result, losses = [], {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]#just 0 or 1
            loss_detection, loss_box_reg = \
                fastrcnn_loss(cls_logits, box_regression,
                                     det_labels, regression_targets)

            loss_reid = self.reid_loss(person_feat, labels)

            losses = dict(loss_detection=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_reid=loss_reid)
        else:
            boxes, scores, person_feat, labels = \
                self.oim_postprocess_detections(cls_logits, box_regression,person_feat,
                                            proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        person_feat=person_feat[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def oim_postprocess_detections(self, class_logits, box_regression, person_feats_, proposals, image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.sigmoid(class_logits)
        num_classes = class_logits.shape[-1]

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_person_feats = person_feats_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_person_feats = []
        for boxes, scores, person_feats, image_shape in zip(pred_boxes, pred_scores, pred_person_feats, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]


            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            person_feats = person_feats.reshape(-1,self.reid_head.feat_dims)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels ,person_feats= boxes[inds], scores[inds], labels[inds] ,person_feats[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels ,person_feats= boxes[keep], scores[keep], labels[keep] ,person_feats[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels ,person_feats= boxes[keep], scores[keep], labels[keep] ,person_feats[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_person_feats.append(person_feats)

        return all_boxes, all_scores, all_person_feats, all_labels
# class PS_RPN(RegionProposalNetwork):
#     def __init__(self,*args,**kwargs):
#         super(PS_RPN,self).__init__(*args,**kwargs)
#     def forward(self,
#                 images,       # type: ImageList
#                 features,     # type: Dict[str, Tensor]
#                 targets=None  # type: Optional[List[Dict[str, Tensor]]]
#                 ):
#         # type: (...) -> Tuple[List[Tensor], Dict[str, Tensor]]
#         """
#         Arguments:
#             images (ImageList): images for which we want to compute the predictions
#             features (OrderedDict[Tensor]): features computed from the images that are
#                 used for computing the predictions. Each tensor in the list
#                 correspond to different feature levels
#             targets (List[Dict[Tensor]]): ground-truth boxes present in the image (optional).
#                 If provided, each element in the dict should contain a field `boxes`,
#                 with the locations of the ground-truth boxes.
#
#         Returns:
#             boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
#                 image.
#             losses (Dict[Tensor]): the losses for the model during training. During
#                 testing, it is an empty dict.
#         """
#         # RPN uses all feature maps that are available
#         features = list(features.values())
#         objectness, pred_bbox_deltas = self.head(features)
#         anchors = self.anchor_generator(images, features)
#
#         num_images = len(anchors)
#         num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
#         num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
#         objectness, pred_bbox_deltas = \
#             concat_box_prediction_layers(objectness, pred_bbox_deltas)
#         # apply pred_bbox_deltas to anchors to obtain the decoded proposals
#         # note that we detach the deltas because Faster R-CNN do not backprop through
#         # the proposals
#         proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
#         proposals = proposals.view(num_images, -1, 4)
#         boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)
#
#         losses = {}
#         import numpy as np
#         det_targets=targets.copy()
#         for i in range(len(targets)):
#             ind = np.where(targets[i]['labels'] != 5555)
#             nind = np.where(targets[i]['labels'] == 5555)
#             det_targets[i]['labels'][ind] = 1
#             det_targets[i]['labels'][nind] = 0
#         if self.training:
#             assert det_targets is not None
#             labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, det_targets)
#             regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
#             loss_objectness, loss_rpn_box_reg = self.compute_loss(
#                 objectness, pred_bbox_deltas, labels, regression_targets)
#             losses = {
#                 "loss_objectness": loss_objectness,
#                 "loss_rpn_box_reg": loss_rpn_box_reg,
#             }
#         return boxes, losses
