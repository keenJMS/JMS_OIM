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

def test_vis(model,thresh=0.7):
    import matplotlib.pyplot as plt
    testset=CUHK_SYSU('/root/dataset', transform=get_transform(True,0.5), test_size=50, mode='test')
    loader = DataLoader(testset, batch_size=1, shuffle=False, collate_fn=testset.collate_fn)
    model.eval()
    model.cuda()
    for i,(img,target) in enumerate(loader):
        img, target = ship_data_to_cuda(img, target, 'cuda')
        result = model(img, target)
        scores=result[0]['scores'].cpu()
        ind_over_thresh=np.where(scores>thresh)
        pre_boxes=result[0]['boxes'][ind_over_thresh]
        img=(img[0]*255).permute([1,2,0]).cpu().numpy().astype(int)
        for box in target[0]['boxes']:
            draw_box_in_image(img,box)
        for box in pre_boxes:
            draw_box_in_image(img,box,False)
        plt.imshow(img)
        plt.show()
if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='2,3'

    model=build_model.build()
    model=torch.load('model.pth')
    opt = torch.optim.Adam(model.parameters(), lr=0.003)

    #model.load_state_dict()
    test_vis(model)
    exit()
    dataset = CUHK_SYSU('/root/dataset', transform=get_transform(True,0.5), test_size=50, mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    model.cuda()

    for epoch in range(1):
        loss_dict = 0
        count = 0
        for img,target in tqdm(loader,ncols=0,miniters=50):

            iterval=50

            for i in range(len(target)):
                ind = np.where(target[i]['labels'] !=5555)
                nind=np.where(target[i]['labels'] ==5555)
                target[i]['labels'][ind]=1
                target[i]['labels'][nind] = 0

            #print('or',target)
            img,target=ship_data_to_cuda(img,target,'cuda')
            result=model(img,target)

            # for item in result:
            #     print('\nepoch:',epoch,item, result[item])
            losses=sum(loss for loss in result.values())

            if (count+1)%iterval==0  :
                print('\nlosses:', loss_dict / iterval)
                loss_dict=0
            else:
                loss_dict += losses
            opt.zero_grad()
            losses.backward()
            opt.step()
            count+=1
    torch.save(model,'model.pth')

