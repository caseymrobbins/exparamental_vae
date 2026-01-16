import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from bhi_vae_mnist import BHiVAE, GateConfig, compute_gates, compute_loss, compute_metrics


@dataclass
class RecommendedHParams:
    latent_dim: int = 16
    batch_size: int = 128
    lr: float = 1e-3
    optimizer: str = "muon"
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    decoder_likelihood: str = "bernoulli"
    d_ok: float = 100.0
    k_min: float = 2.0
    k_max: float = 10.0
    tau_recon: float = 10.0
    tau_kl_low: float = 1.0
    tau_kl_high: float = 1.0
    tau_ink: float = 50.0
    tau_bg: float = 0.1


def write_csv_rows(path: Path, header: Iterable[str], rows: List[Dict[str, float]]) -> None:
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_recon_grid(
    output_dir: Path,
    epoch: int,
    step: int,
    x: torch.Tensor,
    recon: torch.Tensor,
    num_samples: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = x[:num_samples].cpu()
    recon = recon[:num_samples].cpu()
    grid = utils.make_grid(torch.cat([x, recon], dim=0), nrow=num_samples)
    utils.save_image(grid, output_dir / f"recon_epoch{epoch:03d}_step{step:05d}.png")


def save_latent_traversal(
    output_dir: Path,
    epoch: int,
    step: int,
    model: BHiVAE,
    latent_dim: int,
    steps: int,
    dims: int,
    device: torch.device,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base = torch.zeros(1, latent_dim, device=device)
    traversal = []
    values = torch.linspace(-2.5, 2.5, steps, device=device)
    for dim in range(min(dims, latent_dim)):
        z = base.repeat(steps, 1)
        z[:, dim] = values
        with torch.no_grad():
            recon = model.decode(z).sigmoid().cpu()
        traversal.append(recon)
    if traversal:
        grid = utils.make_grid(torch.cat(traversal, dim=0), nrow=steps)
        utils.save_image(grid, output_dir / f"latent_epoch{epoch:03d}_step{step:05d}.png")


def build_dataloaders(data_dir: str, img_size: int, batch_size: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)


def build_optimizer(model: BHiVAE, args: argparse.Namespace) -> torch.optim.Optimizer:
    beta1 = args.momentum if args.momentum is not None else args.beta1
    betas = (beta1, args.beta2)
    if args.optimizer == "muon":
        from optim.muon import Muon

        return Muon(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def log_batch_metrics(
    output_dir: Path,
    epoch: int,
    step: int,
    metrics: Dict[str, torch.Tensor],
    gates: Dict[str, torch.Tensor],
) -> None:
    per_sample_rows = []
    for idx in range(metrics["recon_loss"].shape[0]):
        per_sample_rows.append(
            {
                "epoch": epoch,
                "step": step,
                "sample_idx": idx,
                "recon_loss": metrics["recon_loss"][idx].item(),
                "kl": metrics["kl"][idx].item(),
                "ink": metrics["ink_mass"][idx].item(),
                "bg": metrics["bg_mean"][idx].item(),
                "g_recon": gates["g_recon"][idx].item(),
                "g_KL-low": gates["g_kl_low"][idx].item(),
                "g_kl_high": gates["g_kl_high"][idx].item(),
                "g_ink": gates["g_ink"][idx].item(),
                "g_bg": gates["g_bg"][idx].item(),
            }
        )

    per_sample_header = per_sample_rows[0].keys()
    write_csv_rows(output_dir / "metrics_per_sample.csv", per_sample_header, per_sample_rows)

    batch_row = {
        "epoch": epoch,
        "step": step,
        "recon_loss": metrics["recon_loss"].mean().item(),
        "kl": metrics["kl"].mean().item(),
        "ink": metrics["ink_mass"].mean().item(),
        "bg": metrics["bg_mean"].mean().item(),
        "g_recon": gates["g_recon"].mean().item(),
        "g_KL-low": gates["g_kl_low"].mean().item(),
        "g_kl_high": gates["g_kl_high"].mean().item(),
        "g_ink": gates["g_ink"].mean().item(),
        "g_bg": gates["g_bg"].mean().item(),
    }
    write_csv_rows(output_dir / "metrics_batch.csv", batch_row.keys(), [batch_row])


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = build_dataloaders(args.data_dir, args.img_size, args.batch_size)
    model = BHiVAE(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    optimizer = build_optimizer(model, args)

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

    output_dir = Path(args.output_dir)
    model.train()
    global_step = 0
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
            )
            gates = compute_gates(metrics, gate_cfg)
            loss, _ = compute_loss(gates)
            loss.backward()
            optimizer.step()

            if global_step % args.log_every == 0:
                log_batch_metrics(output_dir, epoch, global_step, metrics, gates)
                recon = metrics["recon"]
                save_recon_grid(output_dir / "recon", epoch, global_step, x, recon, args.num_recon)
                save_latent_traversal(
                    output_dir / "traversal",
                    epoch,
                    global_step,
                    model,
                    args.latent_dim,
                    args.traversal_steps,
                    args.traversal_dims,
                    device,
                )

            global_step += 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MNIST experiment runner with gated metrics")
    parser.add_argument("--data-dir", default="./data", help="MNIST data directory")
    parser.add_argument("--output-dir", default="./runs/exp", help="Where to write logs/images")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("muon", "adam"),
        default="muon",
        help="Optimizer choice (Muon uses Adam-style betas).",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="Optimizer beta1 coefficient.")
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
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-recon", type=int, default=8, help="Samples per recon grid row")
    parser.add_argument("--traversal-steps", type=int, default=8)
    parser.add_argument("--traversal-dims", type=int, default=8)
    parser.add_argument(
        "--print-recommended",
        action="store_true",
        help=(
            "Print recommended hyperparameters for quick copy/paste. "
            "Includes g_recon, g_KL-low, g_ink gate strings for search alignment."
        ),
    )
    return parser


def maybe_print_recommended(args: argparse.Namespace) -> None:
    if not args.print_recommended:
        return
    rec = RecommendedHParams()
    lines = [
        "Recommended hyperparameters (search: g_recon, g_KL-low, g_ink):",
        f"latent_dim={rec.latent_dim}",
        f"batch_size={rec.batch_size}",
        f"lr={rec.lr}",
        f"optimizer={rec.optimizer}",
        f"beta1={rec.beta1}",
        f"beta2={rec.beta2}",
        f"weight_decay={rec.weight_decay}",
        f"decoder_likelihood={rec.decoder_likelihood}",
        f"d_ok={rec.d_ok}",
        f"k_min={rec.k_min}",
        f"k_max={rec.k_max}",
        f"tau_recon={rec.tau_recon}",
        f"tau_kl_low={rec.tau_kl_low}",
        f"tau_kl_high={rec.tau_kl_high}",
        f"tau_ink={rec.tau_ink}",
        f"tau_bg={rec.tau_bg}",
    ]
    print("\n".join(lines))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    maybe_print_recommended(args)
    train(args)


if __name__ == "__main__":
    main()
