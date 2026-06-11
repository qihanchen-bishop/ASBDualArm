"""RSSM world model components for RGB next-frame prediction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    """Latent state of the RSSM."""

    deter: torch.Tensor
    stoch: torch.Tensor


class ConvEncoder(nn.Module):
    """CNN encoder that maps RGB images to compact embeddings."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 1024) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)
        x = x.flatten(start_dim=1)
        return self.proj(x)


class ConvDecoder(nn.Module):
    """CNN decoder that reconstructs RGB image from latent features."""

    def __init__(self, feature_dim: int, out_channels: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(feature_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.fc(features)
        x = x.view(features.shape[0], 256, 4, 4)
        return self.deconv(x)


class RSSM(nn.Module):
    """Recurrent state-space model with Gaussian stochastic latent."""

    def __init__(
        self,
        action_dim: int,
        deter_dim: int = 200,
        stoch_dim: int = 30,
        embed_dim: int = 1024,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.encoder = ConvEncoder(in_channels=3, embed_dim=embed_dim)

        self.action_stoch = nn.Linear(action_dim + stoch_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, deter_dim)

        self.prior = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        self.posterior = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, 2 * stoch_dim),
        )

        self.decoder = ConvDecoder(feature_dim=deter_dim + stoch_dim, out_channels=3)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.stoch_dim, device=device)
        return RSSMState(deter=deter, stoch=stoch)

    def _stats_to_dist(self, stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_param = torch.chunk(stats, chunks=2, dim=-1)
        std = F.softplus(std_param) + 1e-4
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mean + eps * std

    def imagine_step(self, prev_state: RSSMState, action: torch.Tensor) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor]]:
        x = torch.cat([prev_state.stoch, action], dim=-1)
        x = F.elu(self.action_stoch(x))
        deter = self.gru(x, prev_state.deter)

        prior_stats = self.prior(deter)
        prior_mean, prior_std = self._stats_to_dist(prior_stats)
        stoch = self._sample(prior_mean, prior_std)

        return RSSMState(deter=deter, stoch=stoch), (prior_mean, prior_std)

    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        image_embed: torch.Tensor,
    ) -> tuple[RSSMState, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        prior_state, (prior_mean, prior_std) = self.imagine_step(prev_state, action)
        post_in = torch.cat([prior_state.deter, image_embed], dim=-1)
        post_stats = self.posterior(post_in)
        post_mean, post_std = self._stats_to_dist(post_stats)
        stoch = self._sample(post_mean, post_std)
        post_state = RSSMState(deter=prior_state.deter, stoch=stoch)
        return post_state, (prior_mean, prior_std), (post_mean, post_std)

    def get_features(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)

    def reconstruct(self, state: RSSMState) -> torch.Tensor:
        return self.decoder(self.get_features(state))

    def rollout_observe(
        self,
        actions: torch.Tensor,
        images: torch.Tensor,
        initial_state: RSSMState | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run posterior rollout.

        Args:
            actions: (B, T, A)
            images: (B, T, C, H, W)
        """
        batch_size, horizon = actions.shape[:2]
        device = actions.device
        state = initial_state or self.initial_state(batch_size=batch_size, device=device)

        recons = []
        prior_means = []
        prior_stds = []
        post_means = []
        post_stds = []

        for t in range(horizon):
            embed_t = self.encoder(images[:, t])
            state, (prior_mean, prior_std), (post_mean, post_std) = self.observe_step(
                prev_state=state,
                action=actions[:, t],
                image_embed=embed_t,
            )
            recon_t = self.reconstruct(state)
            recons.append(recon_t)
            prior_means.append(prior_mean)
            prior_stds.append(prior_std)
            post_means.append(post_mean)
            post_stds.append(post_std)

        return {
            "recon": torch.stack(recons, dim=1),
            "prior_mean": torch.stack(prior_means, dim=1),
            "prior_std": torch.stack(prior_stds, dim=1),
            "post_mean": torch.stack(post_means, dim=1),
            "post_std": torch.stack(post_stds, dim=1),
        }


def kl_normal(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussian distributions."""
    post_var = post_std.square()
    prior_var = prior_std.square()
    kl = (
        torch.log(prior_std / post_std)
        + (post_var + (post_mean - prior_mean).square()) / (2.0 * prior_var)
        - 0.5
    )
    return kl.sum(dim=-1)
