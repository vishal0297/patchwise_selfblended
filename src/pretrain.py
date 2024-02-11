import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
import sys
import random
from utils.loss import SupConLoss
from utils.sbi import SBI_Dataset
from utils.scheduler import LinearDecayLR
from sklearn.metrics import confusion_matrix, roc_auc_score
import argparse
from utils.logs import log
from utils.funcs import load_json
from datetime import datetime
from tqdm import tqdm
from model import Detector, patchwiseDetector

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

def compute_accuray(pred,true):
    pred_idx=pred.argmax(dim=1).cpu().data.numpy()
    tmp=pred_idx==true.cpu().numpy()
    return sum(tmp)/len(pred_idx)

def main():
    # cfg=load_json(args.config)

    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')


    image_size=256#cfg['image_size']
    batch_size=2#cfg['batch_size']
    train_dataset=SBI_Dataset(phase='train',image_size=image_size)
    # val_dataset=SBI_Dataset(phase='val',image_size=image_size)
   
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size//2,
                        shuffle=True,
                        collate_fn=train_dataset.pretrain_collate_fn,
                        num_workers=4,
                        drop_last=True
                        )
    # val_loader=torch.utils.data.DataLoader(val_dataset,
    #                     batch_size=batch_size,
    #                     shuffle=False,
    #                     collate_fn=val_dataset.collate_fn,
    #                     num_workers=4,
    #                     pin_memory=True,
    #                     worker_init_fn=val_dataset.worker_init_fn
    #                    )
    
    model=patchwiseDetector().cuda() #torch.nn.DataParallel(patchwiseDetector(), device_ids=[0, 1, 2, 3]).cuda()
    
    

    iter_loss=[]
    train_losses=[]
    test_losses=[]
    train_accs=[]
    test_accs=[]
    val_accs=[]
    val_losses=[]
    # n_epoch=cfg['epoch']
    
    # lr_scheduler=LinearDecayLR(optimizer, n_epoch, int(n_epoch/4*3))
    last_loss=99999


    # now=datetime.now()
    # save_path='output/{}_'.format(args.session_name)+now.strftime(os.path.splitext(os.path.basename(args.config))[0])+'_'+now.strftime("%m_%d_%H_%M_%S")+'/'
    # os.mkdir(save_path)
    # os.mkdir(save_path+'pretrained_weights/')
    # os.mkdir(save_path+'pretrain_logs/')
    # logger = log(path=save_path+"logs/", file="losses.logs")

    # criterion= SupConLoss() #nn.CrossEntropyLoss()
    #SAM(parameters(),torch.optim.SGD,lr=0.001, momentum=0.9)

    n_epoch = 10
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train(mode=True)
        for i, data in enumerate(tqdm(train_loader)):
            # print("Step ",i)
            img=data['img'].cuda().float()
            labels=data['label'].cuda().long()
            loss = model.training_step(img,labels)
            #loss = criterion(output,labels)
            train_loss += loss.item()
        print("Epoch: {}, Supervised contrastive loss: {}".format(epoch,train_loss/len(train_loader)))
    torch.save(model.state_dict(),"./models/pretrained_effnet.pth")

if __name__=='__main__':
    main()