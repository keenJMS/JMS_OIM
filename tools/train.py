import sys
print(sys.path)
sys.path.append('./')
print(sys.path)

from modeling import build_model

from data.datasets.loader import get_data_loader
from tools.logger import setup_logger
from tools.util import Color
from config import cfg

from tools.util import ship_data_to_cuda,draw_box_in_image
from torch.optim.lr_scheduler import StepLR
import torch
from tqdm import tqdm
import os
import os.path as osp
import numpy as np
import argparse
import time
from tools.test_oim import test_benchmark
from tools.solver import get_optimizer,get_lr_scheduler
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
def main(cfg,logger):
    #model

    model = build_model.build(cfg)
    cudnn_benchmark = True
    torch.cuda.manual_seed(1)


    model = model.cuda()

    opt= get_optimizer(cfg,model)

    scheduler = get_lr_scheduler(cfg,opt)
    dataloader=get_data_loader(cfg,mode='train')
    iter_time=int(dataloader.dataset.__len__()/dataloader.batch_size)
    writer = SummaryWriter(logger_dir)
    color=[Color('G'),Color('M'),Color('R'),Color('Y'),Color('B')]
    iter_count=0
    for epoch in range(cfg.SOLVER.MAX_EPOCHS):
        loss_dict = 0

        for i,(img,target) in enumerate(dataloader):
            start_time=time.time()
            iterval=100


            img,target=ship_data_to_cuda(img,target,'cuda')
            result=model(img,target)
            result['loss_reid'] = 1 * result['loss_reid']
            losses=sum(loss for loss in result.values())
            keys=list(result.keys())

            if (i+1)%iterval==0  :
                print('\n{} '.format(Color('G')))
                logger.info('epoch:{0},iter:[{1}/{2}],losses:{3},lr:{4},time_per_batch:{5:.3f}'.format(epoch,
                                                                                                       i, iter_time,
                                                                                                       loss_dict / iterval,
                                                                                            opt.param_groups[0]['lr'],
                                                                                            time.time()-start_time) )
                logger.info('loss_detection:{0:.2f};loss_box_reg:{1:.2f};'
                            'loss_reid:{2:.2f};loss_objectness:{3:.2f};loss_rpn_box_reg:{4:.2f};'
                            .format(*[result[key].item() for key in keys]))
                for key in keys:
                    writer.add_scalar('{}'.format(key), result[key].item(), iter_count)
                loss_dict=0
            else:
                loss_dict += losses
            opt.zero_grad()
            losses.backward()
            opt.step()
            iter_count +=1


        scheduler.step()
        save_path=osp.join(logger_dir,'ep{}.pth'.format(epoch))
        torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': opt.state_dict(),
                'lr_scheduler': scheduler.state_dict()
            }, save_path)
        det_rate,det_ap,map,rank1=test_benchmark(save_path,cfg,logger)
        writer.add_scalar('det_rate',det_rate,epoch)
        writer.add_scalar('det_ap',det_ap,epoch)
        writer.add_scalar('map',map,epoch)
        writer.add_scalar('rank1',rank1,epoch)
    writer.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description="person search")
    parser.add_argument("--config_file",default='/root/proj/JMS_OIM/config/baseline.yml')
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.config_file !="":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #logger
    logger_dir=osp.join(cfg.OUTPUT_DIR,time.asctime()[4:])
    if not osp.exists(logger_dir):
        os.mkdir(logger_dir)
    logger= setup_logger("person search",logger_dir,'log.txt'.format(time.asctime()[4:]),0)
    logger.info(cfg)
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    cudnn_benchmark= True
    main(cfg,logger)
