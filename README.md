# airfoil-flow-surrogate

Neural-network surrogates for 2D airfoil CFD: predict the pressure and velocity fields around an airfoil in milliseconds instead of running a RANS simulation. The models are trained and evaluated on the AirfRANS dataset of 1,000 high-fidelity simulations.

> **Status: in development.** The container stack is in place; the data pipeline comes next.

## Planned pipeline

```
AirfRANS Dataset.zip  (1,000 RANS simulations as VTK files, 10 GB)
   │  ingestion: resumable download, SHA-256, archive listing inspected before extraction
   ▼
data/raw/airfrans/<snapshot-date>/                      immutable
   │  ETL: .vtu / .vtp  →  per-simulation parquet
   │       (mesh nodes, mesh edges, model inputs, CFD targets) + quality gate
   ▼
data/interim/  →  data/processed/                        normalisation fitted on training simulations only
   │
   ├── naive free-stream baseline
   ├── MLP            (per-point)
   └── GraphSAGE GNN  (the CFD mesh as a graph)
   ▼
Evaluation on 200 unseen airfoils
   field errors · surface pressure coefficient · lift coefficient
   ▼
Plots + .vtu exports (true / predicted / error) for ParaView
```

## Running it

Everything runs in Docker, so no local Python or CUDA installation is needed. You need Docker Desktop (WSL2 backend on Windows) and an NVIDIA GPU with a current driver. The PyTorch wheels inside the image bring their own CUDA runtime.

```bash
cd servicios
cp .env.example .env    # replace CHANGE_ME with a generated token
docker compose --profile ml build
docker compose --profile ml run --rm ml       # test suite, on the GPU
docker compose --profile dev up -d            # JupyterLab at http://localhost:8889
```

`ml` is a job runner, not a server: every pipeline stage runs as `docker compose --profile ml run --rm ml python -m <module>`. Data is written to `data/` through a bind mount and is never part of an image.

## Data

[AirfRANS](https://github.com/Extrality/airfrans_lib) (Bonnet et al., NeurIPS 2022 Datasets and Benchmarks Track): incompressible RANS solutions over NACA 4- and 5-digit airfoils, Reynolds number 2–6 million, subsonic regime. The dataset is licensed under [ODbL-1.0](https://opendatacommons.org/licenses/odbl/1-0/). It is **not redistributed here**; the pipeline downloads it from the authors' official host.

```bibtex
@inproceedings{
bonnet2022airfrans,
title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier{\textendash}Stokes Solutions},
author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://arxiv.org/abs/2212.07564}
}
```
