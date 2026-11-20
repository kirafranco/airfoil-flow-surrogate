# airfoil-flow-surrogate

Neural-network surrogates for 2D airfoil CFD: predict the pressure and velocity fields around an airfoil in milliseconds instead of running a RANS simulation. The models are trained and evaluated on the AirfRANS dataset of 1,000 high-fidelity simulations.

> **Status: design phase.** This repo currently contains the project brief only. No application code exists yet. Features move through `pending → spec_ready → approval → in_progress → done`.

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
