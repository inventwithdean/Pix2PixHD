import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, lambda1=10, lambda2=10, norm_weight_to_one=True):
        super().__init__()

        lambda0 = 1.0

        # Keeping same ratio but scale down max to 1.0 if norm_weight_to_one is True
        scale = max(lambda0, lambda1, lambda2) if norm_weight_to_one else 1.0

        self.lambda0 = lambda0 / scale
        self.lambda1 = lambda1 / scale
        self.lambda2 = lambda2 / scale

    # Using Least Squared Adversarial Loss for stability
    def adv_loss(self, discriminator_preds, is_real):
        # Using pointers to Torch's ones_like and zeros_like functions
        target = torch.ones_like if is_real else torch.zeros_like

        adv_loss = 0
        for preds in discriminator_preds:
            pred = preds[-1]
            adv_loss += F.mse_loss(pred, target(pred))

        return adv_loss

    # Feature Matching Loss
    def fm_loss(self, real_preds, fake_preds):
        fm_loss = 0
        for real_features, fake_features in zip(real_preds, fake_preds):
            for real_feature, fake_feature in zip(real_features, fake_features):
                fm_loss += F.l1_loss(real_feature.detach(), fake_feature)

        return fm_loss

    def forward(self, x_real, label_map, generator, discriminator):
        x_fake = generator(label_map)
        fake_preds_for_g = discriminator(torch.cat((label_map, x_fake), dim=1))
        fake_preds_for_d = discriminator(torch.cat((label_map, x_fake.detach()), dim=1))
        real_preds_for_d = discriminator(torch.cat((label_map, x_real.detach()), dim=1))

        g_loss = self.lambda0 * self.adv_loss(fake_preds_for_g, True) + (
            self.lambda1
            * self.fm_loss(real_preds_for_d, fake_preds_for_g)
            / discriminator.n_discriminators
        )

        d_loss = 0.5 * (
            self.adv_loss(real_preds_for_d, True)
            + self.adv_loss(fake_preds_for_d, False)
        )

        return g_loss, d_loss, x_fake.detach()
