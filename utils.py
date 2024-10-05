import torch
from PIL import Image
import os
import numpy as np

save_dir = "./output"


@torch.no_grad
def save_images(generated_images, epoch, step, edges=False):
    generated_images = (generated_images + 1) / 2
    for i in range(generated_images.size(0)):
        img = generated_images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        postfix = "sketch" if edges else "generated"
        img.save(
            os.path.join(save_dir, f"epoch_{epoch}_step_{step}_img_{i+1}_{postfix}.png")
        )


def load_checkpoint(models, optimizers, schedulers, epoch, checkpoints_dir):
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers
    generator.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"generator_state_dict_{epoch}"))
    )
    discriminator.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"discriminator_state_dict_{epoch}"))
    )
    g_optimizer.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"g_optimizer_state_dict_{epoch}"))
    )
    d_optimizer.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"d_optimizer_state_dict_{epoch}"))
    )
    g_scheduler.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"g_scheduler_state_dict_{epoch}"))
    )
    d_scheduler.load_state_dict(
        torch.load(os.path.join(checkpoints_dir, f"d_scheduler_state_dict_{epoch}"))
    )


def save_checkpoint(models, optimizers, schedulers, epoch, output_dir):
    generator, discriminator = models
    g_optimizer, d_optimizer = optimizers
    g_scheduler, d_scheduler = schedulers
    torch.save(
        generator.state_dict(),
        os.path.join(output_dir, f"generator_state_dict_{epoch}"),
    )
    torch.save(
        discriminator.state_dict(),
        os.path.join(output_dir, f"discriminator_state_dict_{epoch}"),
    )
    torch.save(
        g_optimizer.state_dict(),
        os.path.join(output_dir, f"g_optimizer_state_dict_{epoch}"),
    )
    torch.save(
        d_optimizer.state_dict(),
        os.path.join(output_dir, f"d_optimizer_state_dict_{epoch}"),
    )
    torch.save(
        g_scheduler.state_dict(),
        os.path.join(output_dir, f"g_scheduler_state_dict_{epoch}"),
    )
    torch.save(
        d_scheduler.state_dict(),
        os.path.join(output_dir, f"d_scheduler_state_dict_{epoch}"),
    )
