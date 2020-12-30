import os
import os.path as osp
import torch
from scipy.io import loadmat
import numpy as np
import sys
sys.path.append('./')
from data.datasets.person_search_dataset import PersonSearchDataset
from tools.evaluator import _compute_iou
from numba import jit
from sklearn.metrics import average_precision_score
import torchvision.transforms as T
class CUHK_SYSU(PersonSearchDataset):
    def __init__(self, root, transform=None,mode='train', test_size=50):
        self.dataset_name = 'CUHK-SYSU'
        self.root=root
        ds_path=osp.join(self.root,self.dataset_name)
        self.ds_path=ds_path

        test_sizes=[50,100,500,1000,2000,4000]
        #assert mode == 'train' or mode == 'test','model should be train or test'
        self.mode = mode
        if self.mode == 'test':
            assert test_size in test_sizes ,"test_size should be one of {}".format(test_sizes)
            self.test_size=test_size

        self.train=loadmat(osp.join(ds_path,'annotation/test/train_test/Train.mat'))['Train'].squeeze()
        self.test=loadmat(osp.join(ds_path,'annotation/test/train_test/TestG{}.mat'.format(test_size)))['TestG{}'.format(test_size)].squeeze()
        #self.test_100=loadmat(osp.join(ds_path,'annotation/test/train_test/TestG100.mat'))['TestG100'].squeeze()
        self.Images=loadmat(osp.join(ds_path,'annotation/Images.mat'))['Img'].squeeze()
        super(CUHK_SYSU, self).__init__(root, transform, mode)
    def get_roidb(self):
        """
        first ,read the Train.mat ,then get two dict, scene_with_boxes{'scene_name':boxes(shape=n*4)},scene_with_pids{'scene_name':pids n*[-1]}

        :return:
        """
        scene_with_boxes={}
        scene_with_pids={}
        for i,(img_name,num_box_in_img,boxes) in enumerate(self.Images):
            img_name=img_name[0]
            boxes=np.array([b[0] for b in boxes[0]])
            boxes=boxes.reshape(-1,4)
            valid_index = np.where((boxes[:,2]>0)&(boxes[:,3]>0))
            #(boxes[:,2],boxes[:,3])=(boxes[:,0]+boxes[:,2],boxes[:,1]+boxes[:,3])
            boxes=boxes[valid_index]
            scene_with_boxes[img_name]=boxes
            scene_with_pids[img_name]=-1*np.ones(boxes.shape[0]).astype(np.int32)
            pass
        if self.mode=='train':
            for i,person in enumerate(self.train):
                person_in_scences=person[0,0][2][0]
                for (scene_name,box,ishard) in person_in_scences:
                    match=[(boxx==box[0]).all() for boxx in scene_with_boxes[scene_name[0]]]
                    match_ind=np.where(match)
                    scene_with_pids[scene_name[0]][match_ind] = i
                    #ind=np.where(scene_with_boxes[scence_name[0]]==box[0])
                    pass
        else:
            print("current use TestG{}".format(self.test_size))
            for i,item in enumerate(self.test):
                scene_name= item['Query'][0,0][0][0]
                query_box=item['Query'][0,0][1]
                query_pid=int(item['Query']['idname'][0,0][0][1:])
                match = [(boxx == query_box).all() for boxx in scene_with_boxes[scene_name]]
                match_ind = np.where(match)
                scene_with_pids[scene_name][match_ind] = query_pid
                gallery=item['Gallery'].squeeze()
                for (scene_name,box,ishard) in gallery:
                    if box.size==0:
                        break
                    match=[(boxx==box[0]).all() for boxx in scene_with_boxes[scene_name[0]]]
                    match_ind=np.where(match)
                    scene_with_pids[scene_name[0]][match_ind] = query_pid
        gt_roidb=[]
        for scene_name in self.imgs:
            boxes = scene_with_boxes[scene_name]
            boxes=boxes.astype(np.int32)
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]  # (x1, y1, x2, y2)
            pids = scene_with_pids[scene_name]
            gt_roidb.append({
                'im_name': scene_name,
                'boxes': boxes.astype(np.int32),
                'gt_pids': pids,
                'flipped': False})
        return gt_roidb

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self.ds_path, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]
        if self.mode in ('test', 'probe'):
            return test
        # all images
        all_imgs = loadmat(
            osp.join(self.ds_path, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        train_set=list(set(all_imgs) - set(test))
        train_set.sort()
        return train_set

    def _adapt_pid_to_cls(self, label_pids, upid=5555):
        """
        convert pid range from (0, N-1) to (1, N), and replace -1 with unlabeled_person_identifier 5555
        """
        label_pids += 1
        label_pids += (label_pids == 0).type(torch.int64) * upid
        return label_pids

    def load_probes(self):
        probes = []
        for item in self.test['Query']:
            im_name = str(item['imname'][0, 0][0])
            roi = item['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]

            probes.append({'im_name': im_name,
                           'boxes': roi[np.newaxis, :],
                           'gt_classes': np.array([1]),
                           # Useless. Can be set to any value.
                           'gt_pids': np.array([-100]),
                           'flipped': False})
        return probes
    @staticmethod

    def search_performance_calc(gallery_set, probe_set,
                                gallery_det, gallery_feat, probe_feat,
                                det_thresh=0.5, gallery_size=100,logger=None):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        det_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                            -1 for using full set
        """
        assert len(gallery_set) == len(gallery_det)
        assert len(gallery_set) == len(gallery_feat)
        assert len(probe_set) == len(probe_feat)

        use_full_set = gallery_size == -1
        fname = 'TestG{}'.format(gallery_size if not use_full_set else 50)
        protoc = loadmat(osp.join(gallery_set.ds_path, 'annotation/test/train_test',
                                  fname + '.mat'))[fname].squeeze()

        # mapping from gallery image to (det, feat)
        gt_roidb = gallery_set.record
        name_to_det_feat = {}
        for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
            name = gt['im_name']
            if det != []:
                scores = det[:, 4].ravel()
                inds = np.where(scores >= det_thresh)[0] #thresh must >0.5
                if len(inds) > 0:
                    gt_boxes = gt['boxes']
                    det_boxes, reID_feat_det = det[inds], feat[inds],
                    box_true = []
                    num_gt, num_det = gt_boxes.shape[0], det_boxes.shape[0]

                    # tag if detection is correct; could be skipped.
                    ious = np.zeros((num_gt, num_det), dtype=np.float32)
                    for i in range(num_gt):
                        for j in range(num_det):
                            ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
                    tfmat = (ious >= 0.5)
                    # for each det, keep only the largest iou of all the gt
                    for j in range(num_det):
                        largest_ind = np.argmax(ious[:, j])
                        for i in range(num_gt):
                            if i != largest_ind:
                                tfmat[i, j] = False
                    # for each gt, keep only the largest iou of all the det
                    for i in range(num_gt):
                        largest_ind = np.argmax(ious[i, :])
                        for j in range(num_det):
                            if j != largest_ind:
                                tfmat[i, j] = False
                    for j in range(num_det):
                        if tfmat[:, j].any():
                            box_true.append(True)
                        else:
                            box_true.append(False)

                    assert len(box_true) == len(det_boxes)
                    name_to_det_feat[name] = (
                        det_boxes, reID_feat_det, np.array(box_true))

        aps = []
        accs = []
        topk = [1, 5, 10]
        ret = {'image_root': gallery_set.data_path, 'results': []}
        for i in range(len(probe_set)):
            y_true, y_score, y_true_box = [], [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = str(protoc['Query'][i]['imname'][0, 0][0])
            probe_roi = protoc['Query'][i][
                'idlocate'][0, 0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol
            for item in protoc['Gallery'][i].squeeze():
                gallery_imname = str(item[0][0])
                # some contain the probe (gt not empty), some not
                gt = item[1][0].astype(np.int32)
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat:
                    continue
                det, feat_g, box_true = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({'img': str(gallery_imname),
                                     'roi': map(float, list(gt))})
                    iou_thresh = min(0.5, (w * h * 1.0) /
                                     ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    box_true = box_true[inds]# sort by similarity
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1   #only one shot in a scene!
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                y_true_box.extend(list(box_true))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                for gallery_imname in gallery_set.imgs:
                    if gallery_imname in tested:
                        continue
                    if gallery_imname not in name_to_det_feat:
                        continue
                    det, feat_g, box_true = name_to_det_feat[gallery_imname]
                    # get L2-normalized feature matrix NxD
                    assert feat_g.size == np.prod(feat_g.shape[:2])
                    feat_g = feat_g.reshape(feat_g.shape[:2])
                    # compute cosine similarities
                    sim = feat_g.dot(feat_p).ravel()
                    # guaranteed no target probe in these gallery images
                    label = np.zeros(len(sim), dtype=np.int32)
                    y_true.extend(list(label))
                    y_score.extend(list(sim))
                    y_true_box.extend(list(box_true))
                    imgs.extend([gallery_imname] * len(sim))
                    rois.extend(list(det))
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            y_true_box = np.asarray(y_true_box)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]# sort all gallery for a query
            y_score = y_score[inds]
            y_true = y_true[inds]
            y_true_box = y_true_box[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])#if it have shot in k sample ,it is 1,else 0
            # 4. Save result for JSON dump
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': map(float, list(probe_roi)),
                         'probe_gt': probe_gt,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': map(float, list(rois[inds[k]])),
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                    'det_correct': int(y_true_box[k]),
                })
            ret['results'].append(new_entry)

        if logger==None:
            print('search ranking:')
            print('  mAP = {:.2%}'.format(np.mean(aps)))
            accs = np.mean(accs, axis=0)
            for i, k in enumerate(topk):
                print('  top-{:2d} = {:.2%}'.format(k, accs[i]))
        else:
            logger.info('search ranking:')
            logger.info('  mAP = {:.2%}'.format(np.mean(aps)))
            accs = np.mean(accs, axis=0)
            for i, k in enumerate(topk):
                logger.info('  top-{:2d} = {:.2%}'.format(k, accs[i]))
        ret['mAP'] = np.mean(aps)
        ret['accs'] = accs

        return ret
    def get_data_path(self):
        return osp.join(self.ds_path,'Image','SSM')
if __name__=='__main__':
    # transform_train=T.Compose(
    #                         [
    #                             T.ToTensor(),
    #                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #                         ]
    # )
    from datasets.transformer import get_transform
    t=get_transform(True,0.5)
    dataset=CUHK_SYSU('/root/dataset',transform=t,test_size=50,mode='probe')
    from torch.utils.data import DataLoader
    loader=DataLoader(dataset,batch_size=1,shuffle=True,collate_fn=dataset.collate_fn)


    def draw_box_in_image(img, box, gt=True, l_w=2):
        [gx_min, gy_min, gx_max, gy_max] = box.int().cpu().numpy().tolist()
        color = 0 if gt else 1
        img[gy_min:gy_max, gx_min:gx_min + l_w, color] = 255
        img[gy_min:gy_max, gx_max:gx_max + l_w, color] = 255
        img[gy_min:gy_min + l_w, gx_min:gx_max, color] = 255
        img[gy_max:gy_max + l_w, gx_min:gx_max, color] = 255
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    for (img, target) in tqdm(loader,ncols=0):
        img=(img[0]*255).permute([1,2,0]).cpu().numpy().astype(int)
        for box in target[0]['boxes']:
            draw_box_in_image(img,box)
        plt.imshow(img)
        plt.show()
    pass


