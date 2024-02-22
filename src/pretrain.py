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
from model import Detector, patchwiseDetector, patchwiseDetectorwithfreq
from pytorch_metric_learning import losses


def main():
    # cfg=load_json(args.config)

    seed=5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:2')


    image_size=256
    batch_size=8
    train_dataset=SBI_Dataset(phase='train',image_size=image_size)
   
    train_loader=torch.utils.data.DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=train_dataset.pretrain_collate_fn,
                        num_workers=4,
                        drop_last=True
                        )
    
    model=patchwiseDetectorwithfreq().to(device) #torch.nn.DataParallel(patchwiseDetector(), device_ids=[0, 1, 2, 3]).cuda()
    

    criterion= losses.NTXentLoss(temperature=0.01) #SupConLoss() #nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    n_epoch = 500
    for epoch in range(n_epoch):
        np.random.seed(seed + epoch)
        train_loss=0.
        train_acc=0.
        model.train(mode=True)
        for i, data in enumerate(tqdm(train_loader)):
            # print("Step ",i)
            img=data['img'].to(device).float()
            labels=data['label'].to(device).long()
            outputs = model(img)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print("Epoch: {}, Supervised contrastive loss: {}".format(epoch,train_loss/len(train_loader)))
        torch.save(model.state_dict(),"./models/pretrained_pcl_effnet.pth")

if __name__=='__main__':
    main()