import os
import logging
from config import Config
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
import losses
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


def setup_logger(log_file):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def get_main_output(restored):
    if isinstance(restored, (list, tuple)):
        return restored[-1]
    return restored

######### Set Seeds ###########
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)
log_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'logs', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)
utils.mkdir(log_dir)

# TensorBoard and logger init
writer = SummaryWriter(log_dir=model_dir)
logger = setup_logger(os.path.join(log_dir, "train.log"))

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR

######### Models ###########
from models.model import SDGformer
model_name = SDGformer.__name__
model_restoration = SDGformer()
model_restoration.cuda()


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    logger.info("Using %d GPUs via DataParallel.", torch.cuda.device_count())

new_lr = opt.OPTIM.LR_INITIAL


optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)
best_ckpt_path = os.path.join(model_dir, f"{model_name}_best.pth")
latest_ckpt_path = os.path.join(model_dir, f"{model_name}_latest.pth")

######### Scheduler ###########
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                        eta_min=opt.OPTIM.LR_MIN)
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = best_ckpt_path if os.path.exists(best_ckpt_path) else utils.get_last_path(model_dir, '_best.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = optimizer.param_groups[0]['lr']
    logger.info("Resuming training from %s", path_chk_rest)
    logger.info("Current learning rate: %.8f", new_lr)

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_edge = losses.EdgeLoss()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=12,
                          drop_last=True, pin_memory=True)

val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=12, drop_last=False,
                        pin_memory=True)

logger.info("Model: %s", model_name)
logger.info("Start Epoch: %d | End Epoch: %d", start_epoch, opt.OPTIM.NUM_EPOCHS)
logger.info("Train Dir: %s", train_dir)
logger.info("Val Dir: %s", val_dir)
logger.info("Logs will be saved to: %s", os.path.join(log_dir, "train.log"))

best_psnr = 0
best_epoch = 0

early_stopping = EarlyStopping(patience=100, min_delta=0., verbose=False)

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0.0
    epoch_loss_char = 0.0
    epoch_loss_edge = 0.0

    model_restoration.train()
    progress = tqdm(train_loader, desc=f"Train {epoch}/{opt.OPTIM.NUM_EPOCHS}", leave=False)
    for data in progress:
        optimizer.zero_grad(set_to_none=True)

        target = data[0].cuda(non_blocking=True)
        input_ = data[1].cuda(non_blocking=True)

        restored = get_main_output(model_restoration(input_))

        # Single-output model: compute full-image loss once per batch.
        loss_char = criterion_char(restored, target)
        loss_edge = criterion_edge(restored, target)
        loss = loss_char + 0.05 * loss_edge

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_loss_char += loss_char.item()
        epoch_loss_edge += loss_edge.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_loss_char = epoch_loss_char / len(train_loader)
    avg_epoch_loss_edge = epoch_loss_edge / len(train_loader)
    current_lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('train/LossTotal', avg_epoch_loss, epoch)
    writer.add_scalar('train/LossChar', avg_epoch_loss_char, epoch)
    writer.add_scalar('train/LossEdge', avg_epoch_loss_edge, epoch)
    writer.add_scalar('train/LearningRate', current_lr, epoch)

    #### Evaluation ####
    psnr_val_rgb = None
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        for data_val in tqdm(val_loader, desc="Valid", leave=False):
            target = data_val[0].cuda(non_blocking=True)
            input_ = data_val[1].cuda(non_blocking=True)

            with torch.no_grad():
                restored = get_main_output(model_restoration(input_))

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item() if psnr_val_rgb else 0.0

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch
            logger.info("Best checkpoint updated at epoch %d (PSNR=%.4f)", epoch, psnr_val_rgb)
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, best_ckpt_path)

        logger.info(
            "Validation | Epoch %d | PSNR %.4f | Best Epoch %d | Best PSNR %.4f",
            epoch, psnr_val_rgb, best_epoch, best_psnr
        )

        # Invoke the early stopping strategy.
        early_stopping(psnr_val_rgb)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered at epoch %d", epoch)
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, latest_ckpt_path)
            break

    if epoch % opt.TRAINING.VAL_SAVE_EVERY == 0:
        epoch_ckpt_path = os.path.join(model_dir, f"{model_name}_epoch_{epoch}.pth")
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, epoch_ckpt_path)

    scheduler.step()
    epoch_time = time.time() - epoch_start_time

    if psnr_val_rgb is None:
        logger.info(
            "Epoch %03d | Time %.2fs | Loss %.4f (Char %.4f, Edge %.4f) | LR %.8f",
            epoch, epoch_time, avg_epoch_loss, avg_epoch_loss_char, avg_epoch_loss_edge, current_lr
        )
    else:
        logger.info(
            "Epoch %03d | Time %.2fs | Loss %.4f (Char %.4f, Edge %.4f) | PSNR %.4f | LR %.8f",
            epoch, epoch_time, avg_epoch_loss, avg_epoch_loss_char, avg_epoch_loss_edge, psnr_val_rgb, current_lr
        )

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, latest_ckpt_path)

# close TensorBoard
writer.close()
