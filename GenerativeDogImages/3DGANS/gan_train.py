import os
import ast 
from dataset import ThreeDatasetTrain
import torch
import torch.optim as optim
from tqdm import tqdm 
from itertools import chain
import torch.nn as nn 
from model_dispather import MODEL_DISPATCHER
import argparse
from distutils.util import strtobool
from datetime import datetime
import random
import torchvision.utils as vutils
import numpy as np
import torch.nn.functional as F
from functools import partial
from utils import init_gan_weights, seed_everything,toggle_grad
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import TensorboardLogger,ProgressBar,PiecewiseLinear, ParamGroupScheduler
from torch.utils.data import Subset
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler
from ignite.handlers import ModelCheckpoint, TerminateOnNan
### Get the parameter from the system 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beta1 = 0.5
EPOCH = int(os.environ.get("EPOCH"))
TRAINING_BATCH_SIZE = int(os.environ.get("TRAINING_BATCH_SIZE"))
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE"))

lr=float(os.environ.get("LR"))
OUT_DIR = "../"
lambda_value = float(os.environ.get("LAMBDA"))


seed_everything(42)

def main():
    print("Device is ",DEVICE)
    ## Initial model
    Generator = MODEL_DISPATCHER['generator']
    Discriminators = MODEL_DISPATCHER['discriminator']
    generator_A2B = Generator().to(DEVICE)
    init_gan_weights(generator_A2B)

    discriminators_B = Discriminators().to(DEVICE)
    init_gan_weights(discriminators_B)

    generator_B2A = Generator().to(DEVICE)
    init_gan_weights(generator_B2A)
    discriminators_A = Discriminators().to(DEVICE)
    init_gan_weights(discriminators_A)
    

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    print("Model loaded !!! ") 

    optimizer_G = optim.Adam(chain(generator_A2B.parameters(), generator_B2A.parameters()), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(chain(discriminators_A.parameters(), discriminators_B.parameters()), lr=lr, betas=(beta1, 0.999))

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
    eval_batch_size = 5

    train_random_indices = [random.randint(0, len(train_dataset) - 1) for _ in range(eval_batch_size)]
    test_random_indices = [random.randint(0, len(valid_dataset) - 1) for _ in range(eval_batch_size)]
    eval_train_dataset = Subset(train_dataset, train_random_indices)
    eval_test_dataset = Subset(train_dataset, test_random_indices)

    eval_train_loader = torch.utils.data.DataLoader(eval_train_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)
    eval_test_loader = torch.utils.data.DataLoader(eval_test_dataset, batch_size=eval_batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)

    print("data get!!!!")


    ## Data

    buffer_size = 50
    fake_a_buffer = []
    fake_b_buffer = []


    def buffer_insert_and_get(buffer, batch):
        output_batch = []
        for b in batch:
            b = b.unsqueeze(0)
            # if buffer is not fully filled:
            if len(buffer) < buffer_size:
                output_batch.append(b)
                buffer.append(b.cpu())
            elif random.uniform(0, 1) > 0.5:
                # Add newly created image into the buffer and put ont from the buffer into the output
                random_index = random.randint(0, buffer_size - 1)            
                output_batch.append(buffer[random_index].clone().to(DEVICE))
                buffer[random_index] = b.cpu()
            else:
                output_batch.append(b)
        return torch.cat(output_batch, dim=0)

    


    def discriminators_forward_pass(discriminators, batch_real, batch_fake, fake_buffer):
        decision_real = discriminators(batch_real)
        batch_fake = buffer_insert_and_get(fake_buffer, batch_fake)
        batch_fake = batch_fake.detach()        
        decision_fake = discriminators(batch_fake)
        return decision_real, decision_fake


    def loss_generator(batch_decision, batch_real, batch_rec, lambda_value):
        # loss gan
        target = torch.ones_like(batch_decision)
        loss_gan = F.mse_loss(batch_decision, target)
        # loss cycle
        loss_cycle = F.l1_loss(batch_rec, batch_real) * lambda_value    
        return loss_gan + loss_cycle


    def loss_discriminator(decision_real, decision_fake): 
        loss = F.mse_loss(decision_fake, torch.zeros_like(decision_fake))
        loss += F.mse_loss(decision_real, torch.ones_like(decision_real))
        return loss


    def update_fn(engine, batch):
        generator_A2B.train()
        generator_B2A.train()
        discriminators_A.train()
        discriminators_B.train()    

        real_a = batch[0].to(DEVICE)
        real_b = batch[1].to(DEVICE)
        
        fake_b = generator_A2B(real_a)
        rec_a = generator_B2A(fake_b)
        fake_a = generator_B2A(real_b)
        rec_b = generator_A2B(fake_a)
        decision_fake_a = discriminators_A(fake_a)
        decision_fake_b = discriminators_B(fake_b)

        # Disable grads computation for the discriminators:
        toggle_grad(discriminators_A, False)
        toggle_grad(discriminators_B, False)    
        
        # Compute loss for generators and update generators
        loss_a2b = loss_generator(decision_fake_b, real_a, rec_a, lambda_value)    

        loss_b2a = loss_generator(decision_fake_a, real_b, rec_b, lambda_value)

        # total generators loss:
        loss_generators = loss_a2b + loss_b2a

        optimizer_G.zero_grad()    
        loss_generators.backward()
        optimizer_G.step()

        decision_fake_a = rec_a = decision_fake_b = rec_b = None
        
        # Enable grads computation for the discriminators:
        toggle_grad(discriminators_A, True)
        toggle_grad(discriminators_B, True)    

        decision_real_a, decision_fake_a = discriminators_forward_pass(discriminators_A, real_a, fake_a, fake_a_buffer)    
        decision_real_b, decision_fake_b = discriminators_forward_pass(discriminators_B, real_b, fake_b, fake_b_buffer)    
        # Compute loss for discriminators and update discriminators
        loss_a = loss_discriminator(decision_real_a, decision_fake_a)

        loss_b = loss_discriminator(decision_real_b, decision_fake_b)
        
        # total discriminators loss:
        loss_discriminators = 0.5 * (loss_a + loss_b)
        
        optimizer_D.zero_grad()
        loss_discriminators.backward()
        optimizer_D.step()
        
        return {
            "loss_generators": loss_generators.item(),
            "loss_generator_a2b": loss_a2b.item(),
            "loss_generator_b2a": loss_b2a.item(),
            "loss_discriminators": loss_discriminators.item(),
            "loss_discriminators_a": loss_a.item(),
            "loss_discriminators_b": loss_b.item(),
        }

    trainer = Engine(update_fn)

    metric_names = [
        'loss_discriminators', 
        'loss_generators', 
        'loss_discriminators_a',
        'loss_discriminators_b',
        'loss_generator_a2b',
        'loss_generator_b2a'    
    ]

    def output_transform(out, name):
        return out[name]

    for name in metric_names:
        RunningAverage(output_transform=partial(output_transform, name=name)).attach(trainer, name)
    

    if not os.path.exists(os.path.join("../","tmp/cycle_gan_checkpoints")):
        os.mkdir(os.path.join("../","tmp/cycle_gan_checkpoints"))

    exp_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_logger = TensorboardLogger(log_dir="../tmp/tb_logs/{}".format(exp_name))

    tb_logger.attach(trainer, 
                    log_handler=OutputHandler('training', metric_names), 
                    event_name=Events.ITERATION_COMPLETED)
    print("Experiment name: ", exp_name)


    def evaluate_fn(engine, batch):
        generator_A2B.eval()
        generator_B2A.eval()   
        with torch.no_grad():
            real_a = batch[0].to(DEVICE)
            real_b = batch[1].to(DEVICE)
            # print(real_a.size(),real_b.size())
            fake_b = generator_A2B(real_a)
            rec_a = generator_B2A(fake_b)

            fake_a = generator_B2A(real_b)
            rec_b = generator_A2B(fake_a)
            
        return {
            'real_a': real_a,
            'real_b': real_b,
            'fake_a': fake_a,
            'fake_b': fake_b,
            'rec_a': rec_a,
            'rec_b': rec_b,        
        }

    evaluator = Engine(evaluate_fn)

    @trainer.on(Events.EPOCH_STARTED)
    def run_evaluation(engine):
        evaluator.run(eval_train_loader)
        evaluator.run(eval_test_loader)


    def log_generated_images(engine, logger, event_name):

        tag = "Train" if engine.state.dataloader == eval_train_loader else "Test"
        output = engine.state.output
        state = trainer.state
        global_step = state.get_event_attrib_value(event_name)

        # create a grid:
        # [real a1, real a2, ...]
        # [fake a1, fake a2, ...]
        # [rec a1, rec a2, ...]
        
        s = output['real_a'].shape[0] # we have 5
        res_a = vutils.make_grid(torch.cat([
            output['real_a'][0][0].unsqueeze(1),
            output['fake_b'][0][0].unsqueeze(1),
            output['rec_a'][0][0].unsqueeze(1),
        ]), padding=2, normalize=True, nrow=8).cpu()

        logger.writer.add_image(tag="{} 3D MRI repair demo (Blured, Repaied, Re_Blured)".format(tag), 
                                img_tensor=res_a, global_step=global_step, dataformats='CHW')

        s = output['real_b'].shape[0]
        res_b = vutils.make_grid(torch.cat([
            output['real_b'][0][0].unsqueeze(1),
            output['fake_a'][0][0].unsqueeze(1),
            output['rec_b'][0][0].unsqueeze(1),
        ]), padding=2, normalize=True, nrow=8).cpu()
        logger.writer.add_image(tag="{} 3D MRI repair demo (Origin, Add_Blured, Re_Origin)".format(tag), 
                                img_tensor=res_b, global_step=global_step, dataformats='CHW')

    
    tb_logger.attach(evaluator,
                    log_handler=log_generated_images, 
                    event_name=Events.COMPLETED)
    milestones_values = [
        (0, lr),
        (100, lr),
        (200, 0.0)
    ]
    gen_lr_scheduler = PiecewiseLinear(optimizer_D, param_name='lr', milestones_values=milestones_values)
    desc_lr_scheduler = PiecewiseLinear(optimizer_G, param_name='lr', milestones_values=milestones_values)

    lr_scheduler = ParamGroupScheduler([gen_lr_scheduler, desc_lr_scheduler], 
                                    names=['gen_lr_scheduler', 'desc_lr_scheduler'])

    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    tb_logger.attach(trainer,
                    log_handler=OptimizerParamsHandler(optimizer_G, "lr"), 
                    event_name=Events.EPOCH_STARTED)


    checkpoint_handler = ModelCheckpoint(dirname="../tmp/cycle_gan_checkpoints",
                                        filename_prefix="checkpoint")

    to_save = {
        "generator_A2B": generator_A2B,
        "discriminators_B": discriminators_B,
        "generator_B2A": generator_B2A,
        "discriminators_A": discriminators_A,
        "optimizer_G": optimizer_G,
        "optimizer_D": optimizer_D,
    }

    trainer.add_event_handler(Events.ITERATION_COMPLETED(every=500), checkpoint_handler, to_save)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


    
    ProgressBar(bar_format="").attach(trainer)
    # Epoch-wise progress bar with display of training losses
    ProgressBar(persist=True, bar_format="").attach(trainer, metric_names='all', 
                                                    event_name=Events.EPOCH_STARTED, closing_event_name=Events.COMPLETED)

    trainer.run(train_loader, max_epochs=EPOCH)

    print("Trainning Done !!!")



if __name__ =="__main__":
    main()

    
    