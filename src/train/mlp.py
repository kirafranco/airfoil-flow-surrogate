"""Train the per-node MLP on the training simulations, choose its epoch on validation.

    python -m src.train.mlp --seed 0

In the container:

    docker compose --profile ml run --rm ml python -m src.train.mlp --seed 0

Writes data/models/<run_id>/ with the weights (model.pt), full-resolution
validation metrics (val_metrics.parquet) and metrics.json: parameters, seed,
dataset fingerprint, code commit, the loss curve and validation results. The
test set is not touched here. It is read once, later, by
`python -m src.evaluate run <run_id> --split test`.

Training is deterministic: the same seed on the same machine gives
bit-identical weights.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import MODELS_ROOT
from src.dataset.airfrans import FEATURES, TARGETS, Dataset, Sample, Simulation, features, sample_points
from src.dataset.targets import TargetScaling
from src.etl.schema import write_json
from src.evaluate.evaluate import Prediction, Predictor, evaluate
from src.ingest.airfrans import git_commit
from src.models.mlp import MLP, MlpConfig

logger = logging.getLogger(__name__)

#: Validation draws its points with a seed far from any training seed.
VAL_SEED_OFFSET = 10_000


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 150
    patience: int = 20
    batch_size: int = 8192
    lr: float = 1e-3
    lr_min: float = 1e-5
    volume_points: int = 32_000
    surface_weight: float = 1.0  # λ in the loss: volume MSE + λ · surface MSE
    #: Decoupled weight decay (AdamW). 0 keeps plain Adam, so earlier runs stay reproducible.
    weight_decay: float = 0.0
    hidden: int = 256
    layers: int = 4
    eval_batch: int = 262_144
    device: str = "cuda"


def set_determinism(seed: int) -> None:
    """Same seed, same machine → same weights. Needs CUBLAS_WORKSPACE_CONFIG on a GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


@dataclass
class DeviceSample:
    features: torch.Tensor  # (N, 7)
    targets: torch.Tensor  # (N, 4), scaled
    surface: torch.Tensor  # (N,) float: 1 on surface nodes

    @property
    def size(self) -> int:
        return int(self.features.shape[0])


def to_device(sample: Sample, scaling: TargetScaling, device: torch.device) -> DeviceSample:
    targets = torch.from_numpy(sample.targets).to(device)
    u_inf = torch.from_numpy(sample.u_inf_per_point.astype(np.float32)).to(device)
    return DeviceSample(
        features=torch.from_numpy(sample.features).to(device),
        targets=scaling.scale(targets, u_inf),
        surface=torch.from_numpy(sample.is_surface).to(device=device, dtype=torch.float32),
    )


def loss_fn(pred: torch.Tensor, target: torch.Tensor, surface: torch.Tensor, surface_weight: float) -> torch.Tensor:
    """Volume MSE + λ · surface MSE, as in the AirfRANS paper (λ = 1).

    Written with masks rather than boolean indexing, so a batch never forces a
    GPU-to-CPU sync to count its surface nodes.
    """
    per_point = ((pred - target) ** 2).mean(dim=1)
    n_surface = surface.sum()
    n_volume = surface.numel() - n_surface
    volume = (per_point * (1 - surface)).sum() / n_volume.clamp(min=1)
    on_surface = (per_point * surface).sum() / n_surface.clamp(min=1)
    return volume + surface_weight * on_surface


@torch.no_grad()
def sample_loss(model: MLP, data: DeviceSample, surface_weight: float, batch: int) -> float:
    """The loss over a whole sample, accumulated exactly rather than as a mean of batch means."""
    model.eval()
    volume_sum = torch.zeros((), dtype=torch.float64, device=data.features.device)
    surface_sum = torch.zeros((), dtype=torch.float64, device=data.features.device)
    for start in range(0, data.size, batch):
        stop = start + batch
        per_point = ((model(data.features[start:stop]) - data.targets[start:stop]) ** 2).mean(dim=1).double()
        surface = data.surface[start:stop].double()
        volume_sum += (per_point * (1 - surface)).sum()
        surface_sum += (per_point * surface).sum()
    n_surface = data.surface.double().sum()
    n_volume = data.size - n_surface
    return float(volume_sum / n_volume.clamp(min=1) + surface_weight * surface_sum / n_surface.clamp(min=1))


def mlp_predictor(
    model: MLP, scaling: TargetScaling, normalisation: dict[str, Any], device: torch.device, batch: int = 262_144
) -> Predictor:
    """Full-mesh prediction in physical units, for the evaluator."""
    model.eval()

    def predict(sim: Simulation) -> Prediction:
        x = torch.from_numpy(features(sim, normalisation)).to(device)
        with torch.no_grad():
            scaled = torch.cat([model(x[start : start + batch]) for start in range(0, len(x), batch)])
        u_inf = torch.full((len(x),), sim.u_inf, dtype=torch.float64, device=device)
        return Prediction(fields=scaling.unscale(scaled.double(), u_inf).cpu().numpy())

    return predict


def load_model(run_dir: Path, device: torch.device) -> tuple[MLP, TargetScaling]:
    checkpoint = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = MLP(MlpConfig(**checkpoint["mlp_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, TargetScaling.from_dict(checkpoint["target_scaling"])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def train(
    config: TrainConfig,
    dataset: Dataset | None = None,
    *,
    models_root: Path = MODELS_ROOT,
    now: Callable[[], datetime] = _utc_now,
) -> Path:
    """Train, keep the best-validation epoch, write the run directory; returns it.

    Deterministic algorithms are switched on for the run and the previous
    setting restored afterwards, so a caller in the same process (a test
    suite, a notebook) isn't left with a global it didn't ask for.
    """
    previous = (torch.are_deterministic_algorithms_enabled(), torch.backends.cudnn.benchmark)
    try:
        return _train(config, dataset, models_root=models_root, now=now)
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark = previous[1]


def _train(
    config: TrainConfig,
    dataset: Dataset | None,
    *,
    models_root: Path,
    now: Callable[[], datetime],
) -> Path:
    dataset = dataset if dataset is not None else Dataset.open()
    set_determinism(config.seed)
    device = torch.device(config.device)
    created = now()
    started = time.perf_counter()

    train_sample = sample_points(dataset, dataset.names("train"), volume_points=config.volume_points, seed=config.seed)
    val_sample = sample_points(
        dataset, dataset.names("val"), volume_points=config.volume_points, seed=config.seed + VAL_SEED_OFFSET
    )
    scaling = TargetScaling.fit(train_sample.targets, train_sample.u_inf_per_point)
    train_data = to_device(train_sample, scaling, device)
    val_data = to_device(val_sample, scaling, device)
    del train_sample, val_sample  # the GPU copies are all training needs
    load_seconds = time.perf_counter() - started
    sizes = {
        "train_points": train_data.size,
        "train_surface_points": int(train_data.surface.sum()),
        "val_points": val_data.size,
    }
    logger.info(
        "sample_ready train_points=%d train_surface=%d val_points=%d seconds=%.1f scaling_mean=%s scaling_std=%s",
        sizes["train_points"],
        sizes["train_surface_points"],
        sizes["val_points"],
        load_seconds,
        [round(v, 4) for v in scaling.mean],
        [round(v, 4) for v in scaling.std],
    )

    model = MLP(MlpConfig(hidden=config.hidden, layers=config.layers)).to(device)
    optimiser = (
        torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        if config.weight_decay
        else torch.optim.Adam(model.parameters(), lr=config.lr)
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs, eta_min=config.lr_min)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    curve: list[dict[str, float]] = []
    best_loss, best_epoch, best_state = math.inf, 0, None
    train_started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        order = torch.randperm(train_data.size, device=device, generator=generator)
        total = torch.zeros((), device=device)
        steps = 0
        for start in range(0, train_data.size, config.batch_size):
            index = order[start : start + config.batch_size]
            loss = loss_fn(
                model(train_data.features[index]), train_data.targets[index], train_data.surface[index], config.surface_weight
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += loss.detach()
            steps += 1
        train_loss = float(total / steps)
        val_loss = sample_loss(model, val_data, config.surface_weight, config.eval_batch)
        lr = schedule.get_last_lr()[0]
        schedule.step()
        seconds = time.perf_counter() - epoch_started
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": seconds})
        logger.info(
            "epoch_done epoch=%d train_loss=%.5f val_loss=%.5f lr=%.2e seconds=%.1f", epoch, train_loss, val_loss, lr, seconds
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= config.patience:
            logger.info("early_stop epoch=%d best_epoch=%d best_val_loss=%.5f", epoch, best_epoch, best_loss)
            break
    train_seconds = time.perf_counter() - train_started

    assert best_state is not None
    model.load_state_dict(best_state)
    run_id = f"mlp-{scaling.method}-s{config.seed}-{created:%Y%m%dT%H%M%SZ}"
    run_dir = models_root / run_id
    run_dir.mkdir(parents=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "mlp_config": model.config.as_dict(),
            "target_scaling": scaling.as_dict(),
            "features": list(FEATURES),
            "targets": list(TARGETS),
        },
        run_dir / "model.pt",
    )
    del train_data, val_data  # free the GPU for full-resolution evaluation

    val_table, val_metrics = evaluate(
        mlp_predictor(model, scaling, dataset.normalisation, device, config.eval_batch), dataset, dataset.names("val")
    )
    val_table.to_parquet(run_dir / "val_metrics.parquet")
    metrics = {
        "run_id": run_id,
        "created_at_utc": created.isoformat(timespec="seconds"),
        "model": {"type": "mlp", **model.config.as_dict(), "parameters": model.parameter_count()},
        "targets": scaling.as_dict(),
        "training": {
            **asdict(config),
            "deterministic": True,
            **sizes,
            "epochs_run": len(curve),
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "load_seconds": round(load_seconds, 1),
            "train_seconds": round(train_seconds, 1),
            "curve": curve,
        },
        "dataset": dataset.fingerprint(),
        "code_commit": git_commit(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "val_metrics": val_metrics,
        "test_metrics": None,
    }
    write_json(run_dir / "metrics.json", metrics)
    logger.info(
        "run_done run_id=%s best_epoch=%d val_cl_mae=%.4f val_cl_spearman=%s train_seconds=%.0f",
        run_id,
        best_epoch,
        val_metrics["cl_mae"],
        val_metrics["cl_spearman"],
        train_seconds,
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args(argv)

    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)sZ %(levelname)s %(name)s %(message)s", stream=sys.stdout
    )
    run_dir = train(TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()}))
    logger.info("train_result run_dir=%s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
