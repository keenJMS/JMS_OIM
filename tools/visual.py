import sys
print(sys.path)
sys.path.append('./')
print(sys.path)

from modeling import build_model
from data.datasets.cuhk_sysu import CUHK_SYSU
from data.datasets.transformer import get_transform
from tools.util import ship_data_to_cuda,draw_box_in_image
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
def visual(model_path,cfg=None):
    # model=build_model.build(cfg)
    # checkpoint=torch.load(model_path)
    # model.load_state_dict(checkpoint.state_dict())
    model=torch.load(model_path)
    model=model.cuda()
    model.eval()
    testset=CUHK_SYSU('/root/dataset', transform=get_transform(True,0.5), test_size=50, mode='test')
    loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=testset.collate_fn)

    for i,(img,target) in enumerate(loader):
        img, target = ship_data_to_cuda(img, target, 'cuda')
        result = model(img, target)
        scores=result[0]['scores'].cpu()
        ind_over_thresh=np.where(scores>0)
        pre_boxes=result[0]['boxes'][ind_over_thresh]
        img=(img[0]*255).permute([1,2,0]).cpu().numpy().astype(int)
        for box in target[0]['boxes']:
            draw_box_in_image(img,box)
        for box in pre_boxes:
            draw_box_in_image(img,box,False)
        plt.imshow(img)
        plt.show()
        plt.ioff()
if __name__=='__main__':
    visual('/root/proj/JMS_OIM/outputs/Dec 13 10:59:13 2020/ep16.pth')