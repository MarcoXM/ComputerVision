import os
import ast 
from dataset import ThreeDatasetTrain
import torch
import torch.optim as optim
from tqdm import tqdm 
import torch.nn as nn 
from model_dispather import MODEL_DISPATCHER
import argparse
from distutils.util import strtobool
from datetime import datetime
import os
import random
import numpy as np
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events
from numpy.random.mtrand import RandomState
from torch.utils.data.dataloader import DataLoader
from metrics import EpochMetric,get_psnr
from ignite.handlers import ModelCheckpoint, global_step_from_engine, EarlyStopping, TerminateOnNan
from utils import create_evaluator,create_trainer,LogReport,ModelSnapshotHandler,output_transform,get_lr_scheduler,get_optimizer,seed_everything

### Get the parameter from the system 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_ONE=float(os.environ.get("WEIGHT_ONE"))
WEIGHT_TWO=float(os.environ.get("WEIGHT_TWO"))

EPOCH = int(os.environ.get("EPOCH"))
TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))


BASE_MODEL = os.environ.get("BASE_MODEL")
OUT_DIR = "../"



seed_everything(42)

def main():
    print("Device is ",DEVICE)
    ## Initial model
    model = MODEL_DISPATCHER[BASE_MODEL](16)
    model.to(DEVICE)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    print("Model loaded !!! ") 

    optimizer = get_optimizer(model,0.1,0.9,5e-4,True)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.975)
    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(os.path.join("../",BASE_MODEL)):
        os.mkdir(os.path.join("../",BASE_MODEL))
    OUT_DIR = os.path.join("../",BASE_MODEL,exp_name)
    print("This Exp would be save in ",OUT_DIR)
    os.mkdir(OUT_DIR)
    os.mkdir(os.path.join(OUT_DIR,"weights"))
    os.mkdir(os.path.join(OUT_DIR,"log"))

    ## Data
    train_blur=np.expand_dims(np.load('/u/erdos/students/twang134/three_kernal/dataset/blur.npy'), axis=1)
    train_ground_truth=np.expand_dims(np.load('/u/erdos/students/twang134/three_kernal/dataset/original.npy'), axis=1)

    test_blur=np.expand_dims(np.load('/u/erdos/students/twang134/three_kernal/dataset/validation/blur.npy'), axis=1)
    test_ground_truth=np.expand_dims(np.load('/u/erdos/students/twang134/three_kernal/dataset/validation/original.npy'), axis=1)

    train_dataset = ThreeDatasetTrain(train_blur,train_ground_truth)
    valid_dataset = ThreeDatasetTrain(test_blur,test_ground_truth)

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size=TRAINING_BATCH_SIZE,
        shuffle=True,pin_memory=True,
        num_workers=4,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print("data get!!!!")

    ## Define Trainer
    trainer = create_trainer(model, optimizer, DEVICE,WEIGHT_ONE,WEIGHT_TWO)

    # Recall for Training
    EpochMetric(
        compute_fn=get_psnr,
        output_transform=output_transform
    ).attach(trainer, 'psnr')
    
    pbar = ProgressBar()
    pbar.attach(trainer, metric_names='all')

    evaluator = create_evaluator(model, DEVICE,WEIGHT_ONE,WEIGHT_TWO)
    EpochMetric(
        compute_fn=get_psnr,
        output_transform=output_transform
    ).attach(evaluator, 'psnr')

    def run_evaluator(engine):
        evaluator.run(valid_loader)


    def get_curr_lr(engine):
        lr = lr_scheduler.optimizer.param_groups[0]['lr']
        log_report.report('lr', lr)

    def score_fn(engine):
        score = engine.state.metrics['loss']
        return score

    
    es_handler = EarlyStopping(patience=30, score_function=score_fn, trainer=trainer)
    evaluator.add_event_handler(Events.COMPLETED, es_handler)


    def save_score_fn(engine):
        score = engine.state.metrics['psnr']
        return score

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step())
    best_model_handler = ModelCheckpoint(dirname=os.path.join(OUT_DIR,"weights"),
                                        filename_prefix=f"best_{BASE_MODEL}",
                                        n_saved=3,
                                        global_step_transform=global_step_from_engine(trainer),
                                        score_name="psnr",
                                        score_function=save_score_fn)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {"model": model, })
        
    trainer.add_event_handler(Events.EPOCH_COMPLETED, run_evaluator)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, get_curr_lr)
    
    log_report = LogReport(evaluator, os.path.join(OUT_DIR,"log"))

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_report)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelSnapshotHandler(model, filepath=os.path.join(OUT_DIR,"weights","{}.pth".format(BASE_MODEL))))

    # Clear cuda cache between training/testing
    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    trainer.run(train_loader, max_epochs=EPOCH)


    train_history = log_report.get_dataframe()
    train_history.to_csv(os.path.join(OUT_DIR,"log","{}_log.csv".format(BASE_MODEL)), index=False)

    print(train_history.head())
    print("Trainning Done !!!")



if __name__ =="__main__":
    main()

    
    