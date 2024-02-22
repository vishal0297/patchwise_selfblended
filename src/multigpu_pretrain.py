import torch
from utils.loss import SupConLoss
from utils.sbi import SBI_Dataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.scheduler import LinearDecayLR
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from model import Detector, patchwiseDetector
from tqdm import tqdm
from pytorch_metric_learning import losses

os.environ['NCCL_DEBUG'] = 'INFO'
os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

def ddp_setup():
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    
    # init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # torch.cuda.set_device(rank)
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        loss,
        n_epoch:int
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5) #torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, 1, eta_min=1e-5, last_epoch=-1)
        self.save_every = save_every
        self.model = DDP(model, device_ids=[self.gpu_id],find_unused_parameters=True)
        self.loss_func = loss
        self.max_epoch = n_epoch

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_func(output, targets)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item() 

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)))
        train_loss = 0.0
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for data in tqdm(self.train_data):
            source = data['img'].to(self.gpu_id)
            targets = data['label'].to(self.gpu_id)
            loss = self._run_batch(source, targets)
            train_loss += loss
        print(f"Supervised Contrastive Loss = {train_loss/len(self.train_data)}")

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "pretrained_pcl_steplr_t0.05.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self):
        for epoch in range(self.max_epoch):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs(image_size = 256):
    train_set=SBI_Dataset(phase='train',image_size=image_size)
    model = patchwiseDetector()#torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    train_sampler = DistributedSampler(dataset,shuffle=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.pretrain_collate_fn,
        sampler=train_sampler,
        drop_last=True
    )


def main(save_every: int, total_epochs: int):
    ddp_setup()
    batch_size = 8
    dataset, model, optimizer = load_train_objs()
    loss = losses.NTXentLoss(temperature=0.05) #SupConLoss()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every,loss,total_epochs)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import sys
    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    main(save_every, total_epochs)