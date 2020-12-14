import sys

sys.path.append('./')
from tools.evaluator import inference,detection_performance_calc

import argparse
from config import cfg
from tools.logger import setup_logger
from data.datasets.loader import get_data_loader
import os
import torch
from tools.util import Color
from modeling import build_model

def test_benchmark(model_path,cfg,logger=None):
    model=build_model.build(cfg)

    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model=model.cuda()
    model.eval()
    model_name=model_path.split('/')[-1]
    dataset_name= cfg.DATASETS.NAMES
    if logger == None:
        logger= setup_logger("person search",cfg.OUTPUT_DIR,'TEST-{}.txt'.format(model_name),0)
    logger.info(cfg)
    print(Color('B')+'')
    logger.info("start test")
    logger.info("dataset:{0},model_path:{1}".format(dataset_name,model_path))
    gallery_loader = get_data_loader(cfg,mode='test')
    query_loader = get_data_loader(cfg,mode='probe')
    imgnames_with_boxes,boxes_feats,probe_feats=inference(model,gallery_loader,query_loader,'cuda')
    precision , recall,det_rate,det_ap = detection_performance_calc(gallery_loader.dataset,imgnames_with_boxes.values(),det_thresh=0.01, logger = logger)
    ret= gallery_loader.dataset.search_performance_calc(gallery_loader.dataset,query_loader.dataset,
                                                        imgnames_with_boxes.values(),boxes_feats,probe_feats,
                                                        det_thresh=0.5,gallery_size=gallery_loader.dataset.test_size,logger=logger)

    return det_rate,det_ap,ret['mAP'],ret['accs'][0]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description="person search")
    parser.add_argument("--config_file",default='/root/proj/JMS_OIM/config/baseline.yml')
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.config_file !="":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    cudnn_benchmark= True
    test_benchmark('/root/proj/JMS_OIM/outputs/Dec 14 05:14:45 2020/ep0.pth',cfg)

