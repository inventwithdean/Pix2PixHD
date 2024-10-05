import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import Loss
from dataset import FaceDataset
from global_generator import GlobalGenerator
from pix2pixhd_generator import Pix2PixHDGenerator
from multiscale_discriminator import MultiscaleDiscriminator
from utils import save_images
from utils import load_checkpoint, save_checkpoint
import os

device = "cuda"
epochs = 200
train_dir = "dataset/train"
decay_after = 100
lr = 0.0002
betas = (0.5, 0.999)


def lr_lambda(epoch):
    return (
        1.0
        if epoch < decay_after
        else 1 - float(epoch - decay_after) / (epochs - decay_after)
    )


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)


loss_fn = Loss()

# Phase 1 : Low Resolution (512, 512)

face_dataset1 = FaceDataset("./data/faces", "./data/sketches", target_width=512)
dataloader1 = DataLoader(face_dataset1, batch_size=1, shuffle=True)

generator1 = GlobalGenerator(3, 3).to(device).apply(weights_init)
discriminator1 = (
    MultiscaleDiscriminator(6, n_discriminators=2).to(device).apply(weights_init)
)

g1_optimizer = torch.optim.Adam(list(generator1.parameters()), lr=lr, betas=betas)
d1_optimizer = torch.optim.Adam(list(discriminator1.parameters()), lr=lr, betas=betas)
g1_scheduler = torch.optim.lr_scheduler.LambdaLR(g1_optimizer, lr_lambda)
d1_scheduler = torch.optim.lr_scheduler.LambdaLR(d1_optimizer, lr_lambda)


# Loading Checkpoints
load_epoch = 0
load_checkpoint(
    [generator1, discriminator1],
    [g1_optimizer, d1_optimizer],
    [g1_scheduler, d1_scheduler],
    load_epoch,
    "./global_checkpoints",
)

# Phase 2: High Resolution (1024, 1024)

face_dataset2 = FaceDataset("./data/faces", "./data/sketches", target_width=1024)
dataloader2 = DataLoader(face_dataset1, batch_size=1, shuffle=True)

generator2 = Pix2PixHDGenerator(3, 3).to(device).apply(weights_init)
discriminator2 = MultiscaleDiscriminator(6).to(device).apply(weights_init)

g2_optimizer = torch.optim.Adam(list(generator2.parameters()), lr=lr, betas=betas)
d2_optimizer = torch.optim.Adam(list(discriminator2.parameters()), lr=lr, betas=betas)
g2_scheduler = torch.optim.lr_scheduler.LambdaLR(g2_optimizer, lr_lambda)
d2_scheduler = torch.optim.lr_scheduler.LambdaLR(d2_optimizer, lr_lambda)


def train(
    dataloader, models, optimizers, schedulers, device, checkpoints_dir, load_epoch
):
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers
    cur_step = (load_epoch + 1) * 30000
    display_step = 100
    mean_g_loss = 0.0
    mean_d_loss = 0.0

    for epoch in range(load_epoch + 1, epochs):
        for x_real, sketches in dataloader:
            x_real = x_real.to(device)
            sketches = sketches.to(device)

            g_loss, d_loss, x_fake = loss_fn(x_real, sketches, generator, discriminator)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            mean_g_loss += g_loss.item() / display_step
            mean_d_loss += d_loss.item() / display_step

            if cur_step % display_step == 0 and cur_step > 0:
                print(
                    f"Step: {cur_step}: Generator Loss: {mean_g_loss:.5f}, Discriminator Loss: {mean_d_loss:.5f}"
                )
                save_images(sketches.detach(), epoch, cur_step, True)
                save_images(x_fake.detach(), epoch, cur_step)
                mean_g_loss = 0
                mean_d_loss = 0
            cur_step += 1
        g_scheduler.step()
        d_scheduler.step()
        save_checkpoint(models, optimizers, schedulers, epoch, checkpoints_dir)


# Phase 1 : Low Resolution

train(
    dataloader1,
    [generator1, discriminator1],
    [g1_optimizer, d1_optimizer],
    [g1_scheduler, d1_scheduler],
    device,
    checkpoints_dir="global_checkpoints",
    load_epoch=load_epoch,
)

# Phase 2 : High Resolution

# Using trained global generator with Local Enhancer
generator2.g1 = generator1.g1

train(
    dataloader2,
    [generator2, discriminator2],
    [g2_optimizer, d2_optimizer],
    [g2_scheduler, d2_scheduler],
    device,
    checkpoints_dir="enhanced_checkpoints",
    load_epoch=load_epoch,
)
