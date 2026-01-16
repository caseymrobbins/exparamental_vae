import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from optim.muon import Muon

@dataclass
class GateConfig:
    d_ok: float
    k_min: float
    k_max: float
    tau_recon: float
    tau_kl_low: float
    tau_kl_high: float
    tau_ink: float
    tau_bg: float


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding=1
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.deconv(x)))


class BHiVAE(nn.Module):
    def __init__(self, latent_dim: int, img_size: int, base_channels: int = 32):
        super().__init__()
        if img_size not in (28, 64):
            raise ValueError("img_size must be 28 or 64")
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        if img_size == 28:
            encoder_blocks = [
                ConvBlock(1, base_channels),
                ConvBlock(base_channels, base_channels * 2),
            ]
            self.enc_out_shape = (base_channels * 2, 7, 7)
        else:
            encoder_blocks = [
                ConvBlock(1, base_channels),
                ConvBlock(base_channels, base_channels * 2),
                ConvBlock(base_channels * 2, base_channels * 4),
                ConvBlock(base_channels * 4, base_channels * 4),
            ]
            self.enc_out_shape = (base_channels * 4, 4, 4)

        self.encoder = nn.Sequential(*encoder_blocks)
        enc_out_features = self.enc_out_shape[0] * self.enc_out_shape[1] * self.enc_out_shape[2]
        self.fc_mu = nn.Linear(enc_out_features, latent_dim)
        self.fc_logvar = nn.Linear(enc_out_features, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, enc_out_features)
        if img_size == 28:
            decoder_blocks = [
                DeconvBlock(base_channels * 2, base_channels),
                DeconvBlock(base_channels, base_channels),
            ]
            self.decoder = nn.Sequential(*decoder_blocks)
            self.final = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        else:
            decoder_blocks = [
                DeconvBlock(base_channels * 4, base_channels * 4),
                DeconvBlock(base_channels * 4, base_channels * 2),
                DeconvBlock(base_channels * 2, base_channels),
                DeconvBlock(base_channels, base_channels),
            ]
            self.decoder = nn.Sequential(*decoder_blocks)
            self.final = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(z.size(0), *self.enc_out_shape)
        h = self.decoder(h)
        return self.final(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class FrozenClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def compute_metrics(
    x: torch.Tensor,
    recon_logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    decoder_likelihood: str,
    classifier: Optional[nn.Module] = None,
    labels: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    if decoder_likelihood == "bernoulli":
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_logits, x, reduction="none"
        ).flatten(1).sum(dim=1)
        recon = torch.sigmoid(recon_logits)
    elif decoder_likelihood == "gaussian":
        recon = recon_logits
        recon_loss = F.mse_loss(recon_logits, x, reduction="none").flatten(1).sum(dim=1)
    else:
        raise ValueError("decoder_likelihood must be 'bernoulli' or 'gaussian'")

    kl = 0.5 * (torch.exp(logvar) + mu**2 - 1.0 - logvar).sum(dim=1)
    ink_mass = x.flatten(1).sum(dim=1)
    background_mask = (x < 0.1).float()
    bg_mean = (x * background_mask).flatten(1).sum(dim=1) / (
        background_mask.flatten(1).sum(dim=1) + 1e-6
    )

    metrics = {
        "recon_loss": recon_loss,
        "kl": kl,
        "ink_mass": ink_mass,
        "bg_mean": bg_mean,
        "recon": recon,
    }

    if classifier is not None and labels is not None:
        with torch.no_grad():
            logits = classifier(recon)
            log_probs = F.log_softmax(logits, dim=1)
            class_nll = F.nll_loss(log_probs, labels, reduction="none")
            preds = torch.argmax(logits, dim=1)
            class_acc = (preds == labels).float()
        metrics["class_nll"] = class_nll
        metrics["class_acc"] = class_acc

    return metrics


def compute_gates(metrics: Dict[str, torch.Tensor], gate_cfg: GateConfig) -> Dict[str, torch.Tensor]:
    g_recon = torch.exp(-(metrics["recon_loss"] - gate_cfg.d_ok) / gate_cfg.tau_recon)
    g_kl_low = torch.exp(-(gate_cfg.k_min - metrics["kl"]) / gate_cfg.tau_kl_low)
    g_kl_high = torch.exp(-(metrics["kl"] - gate_cfg.k_max) / gate_cfg.tau_kl_high)
    g_ink = torch.exp(-metrics["ink_mass"] / gate_cfg.tau_ink)
    g_bg = torch.exp(-metrics["bg_mean"] / gate_cfg.tau_bg)
    return {
        "g_recon": g_recon,
        "g_kl_low": g_kl_low,
        "g_kl_high": g_kl_high,
        "g_ink": g_ink,
        "g_bg": g_bg,
    }


def compute_loss(gates: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_stack = torch.stack(
        [gates["g_recon"], gates["g_kl_low"], gates["g_kl_high"], gates["g_ink"], gates["g_bg"]],
        dim=0,
    )
    m = torch.min(gate_stack, dim=0).values
    r = torch.log(m)
    loss = (-r).mean()
    return loss, r


def load_classifier(path: str, device: torch.device) -> nn.Module:
    classifier = FrozenClassifier()
    classifier.load_state_dict(torch.load(path, map_location=device))
    classifier.to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False
    return classifier


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ]
    )
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = BHiVAE(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    beta1 = args.momentum if args.momentum is not None else args.beta1
    betas = (beta1, args.beta2)
    if args.optimizer == "muon":
        optimizer = Muon(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    classifier = None
    if args.classifier_path:
        classifier = load_classifier(args.classifier_path, device)

    gate_cfg = GateConfig(
        d_ok=args.d_ok,
        k_min=args.k_min,
        k_max=args.k_max,
        tau_recon=args.tau_recon,
        tau_kl_low=args.tau_kl_low,
        tau_kl_high=args.tau_kl_high,
        tau_ink=args.tau_ink,
        tau_bg=args.tau_bg,
    )

    model.train()
    for epoch in range(args.epochs):
        for step, (x, labels) in enumerate(train_loader):
            x = x.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(x)
            metrics = compute_metrics(
                x,
                recon_logits,
                mu,
                logvar,
                decoder_likelihood=args.decoder_likelihood,
                classifier=classifier,
                labels=labels,
            )
            gates = compute_gates(metrics, gate_cfg)
            loss, r = compute_loss(gates)
            loss.backward()
            optimizer.step()

            if step % args.log_every == 0:
                message = (
                    f"epoch={epoch} step={step} optimizer={args.optimizer} "
                    f"loss={loss.item():.4f} "
                    f"recon={metrics['recon_loss'].mean().item():.3f} "
                    f"kl={metrics['kl'].mean().item():.3f} "
                    f"ink={metrics['ink_mass'].mean().item():.3f} "
                    f"bg={metrics['bg_mean'].mean().item():.3f} "
                    f"R={r.mean().item():.3f}"
                )
                if "class_nll" in metrics:
                    message += (
                        f" class_nll={metrics['class_nll'].mean().item():.3f} "
                        f"class_acc={metrics['class_acc'].mean().item():.3f}"
                    )
                print(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BHiVAE MNIST training entry point")
    parser.add_argument("--data-dir", default="./data", help="MNIST data directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("muon", "adam"),
        default="muon",
        help="Optimizer choice (Muon uses Adam-style betas).",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Optimizer beta1 coefficient (momentum term).",
    )
    parser.add_argument("--beta2", type=float, default=0.999, help="Optimizer beta2 coefficient.")
    parser.add_argument(
        "--momentum",
        type=float,
        default=None,
        help="Optional momentum override for beta1.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay.")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--img-size", type=int, choices=(28, 64), default=28)
    parser.add_argument("--decoder-likelihood", choices=("bernoulli", "gaussian"), default="bernoulli")
    parser.add_argument("--d-ok", type=float, default=100.0, dest="d_ok")
    parser.add_argument("--k-min", type=float, default=2.0, dest="k_min")
    parser.add_argument("--k-max", type=float, default=10.0, dest="k_max")
    parser.add_argument("--tau-recon", type=float, default=10.0)
    parser.add_argument("--tau-kl-low", type=float, default=1.0)
    parser.add_argument("--tau-kl-high", type=float, default=1.0)
    parser.add_argument("--tau-ink", type=float, default=50.0)
    parser.add_argument("--tau-bg", type=float, default=0.1)
    parser.add_argument("--classifier-path", default=None)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
