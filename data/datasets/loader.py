from torch.utils.data import DataLoader
from data.datasets.cuhk_sysu import CUHK_SYSU
from data.datasets.transformer import get_transform
def get_data_loader(cfg,mode):
    dataset = CUHK_SYSU(cfg.DATASETS.ROOT_DIR, transform=get_transform(mode,0.5), test_size=50, mode=mode)
    loader = DataLoader(dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True if mode=='train' else False, collate_fn=dataset.collate_fn
                        ,num_workers=cfg.DATALOADER.NUM_WORKERS)
    return loader