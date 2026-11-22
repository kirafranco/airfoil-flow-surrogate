# airfoil-flow-surrogate

Neural-network surrogates for 2D airfoil CFD. Given an airfoil's shape and the incoming flow, a model predicts the full pressure and velocity field around it in under a tenth of a second, instead of running a RANS simulation. The models are trained and evaluated on [AirfRANS](#data), a dataset of 1,000 high-fidelity simulations. Every result below is measured on 200 airfoils the models never saw.

> **Status: in active development.** The data pipeline, the evaluator, the baselines and an MLP surrogate are built and tested. Next come a graph neural network on the CFD mesh, validation plots with ParaView exports, and final packaging. They're tracked in [issues](https://github.com/kirafranco/airfoil-flow-surrogate/issues).

## Results so far

The test set is 200 unseen airfoils, scored on every node of every mesh (155,000–210,000 nodes per simulation).

| Model | Relative error: u_x / u_y / p / ν_t | Lift C_L, mean abs. error | Lift ranking, ρ |
|---|---|---|---|
| Free stream (no model) | 0.66 / 0.92 / 1.00 / 1.00 | 0.750 | — |
| Thin-airfoil theory, C_L = 2πα | — | 0.214 | 0.933 |
| MLP 4×256 (mean of 3 seeds) | 0.19 / 0.37 / 0.44 / 0.43 | **0.066** ± 0.011 | **0.993** |
| MLP 4×64, tuned on validation (mean of 3 seeds) | **0.16 / 0.35 / 0.41 / 0.41** | 0.087 ± 0.045 | 0.990 |

**How to read the table:**
- **Relative error** is the relative L2 error of each field over the whole mesh: 0 is perfect.
- The fields are u_x and u_y (velocity), p (pressure) and ν_t (turbulent viscosity).
- **Lift** is integrated from the predicted surface pressure, the same way it is computed from the CFD.
- **ρ** is the Spearman rank correlation between predicted and true lift over the 200 airfoils. It measures whether a model puts the designs in the right order.

**What the numbers show:**
- **Both networks beat both baselines.** The 4×256 MLP's lift error is a third of thin-airfoil theory's. Its lift ranking (0.993) matches the AirfRANS paper's MLP (0.9932), although the paper's forces include wall shear and ours are pressure only.
- **Tuning on validation helped the fields but not lift.** Model selection used the point-wise training loss, which doesn't capture an integrated quantity like lift: one of the tuned model's three seeds gets lift clearly wrong. The graph network will be selected with validation lift in view.
- **The flow fields are still well behind the paper's,** in the paper's standardised MSE: 5–15× for velocity and ν_t, 31–54× for pressure. The likeliest cause is training length: the paper trains for 400 epochs and draws new mesh points every epoch.

**Speed:** predicting a full mesh takes under 0.1 s on a laptop GPU (RTX 3060, 6 GB). Training one model takes 4–9 minutes.

`python -m src.evaluate report` regenerates the full table, which adds columns for surface pressure coefficient, standardised MSE and the paper's rows.

## Pipeline

```
AirfRANS Dataset.zip  (1,000 RANS simulations as VTK files, 10 GB)
   │  ingestion: resumable download, SHA-256, archive listing checked before extraction
   ▼
data/raw/airfrans/<snapshot-date>/                        immutable snapshot
   │  ETL: .vtu / .vtp  →  per-simulation parquet, 10-check quality gate
   ▼
data/interim/     mesh nodes, mesh edges, airfoil surface
data/processed/   splits; normalisation fitted on training simulations only
   │
   ├── free-stream and thin-airfoil baselines             done
   ├── MLP, one prediction per mesh node                  done
   └── GraphSAGE on the mesh graph                        next
   ▼
One evaluator for every model: fields, surface pressure, lift
   ▼
Plots + .vtu exports (true / predicted / error) for ParaView     planned
```

### Data engineering

- **Ingestion:**
  - The download is resumable, retries with exponential backoff, and is guarded against the source file changing between attempts.
  - The archive's SHA-256 is recorded with the snapshot.
  - Nothing is extracted until the archive's file listing passes an allowlist: only `.vtu`, `.vtp` and `.json` files, with no absolute paths, no `..` and no links.
  - Re-running changes nothing.
- **ETL (VTK → parquet):**
  - Each simulation's volume mesh and airfoil surface are read with Kitware's VTK, the library ParaView is built on.
  - Graph edges are taken from the mesh cells, and the stored surface normals are flipped to point outward.
  - The tables are written per simulation, typed and atomically, so a crash never leaves a half-written file.
  - A 10-check quality gate runs on every simulation. It checks for finite values, unique nodes, edges pointing at real nodes, surface points lying on the mesh, and an inflow that agrees with the simulation's name, among others.
  - All 1,000 simulations pass. A full run takes 11 minutes on 2 workers, and a re-run takes 5 seconds and rewrites nothing.
- **Splits are by simulation, never by mesh point.** A point split would leak test geometry into training. The dataset's own manifest gives 800 training and 200 test simulations. 80 of the training simulations are held out for validation with a fixed seed, leaving 720 for training. Normalisation statistics come from those 720 only.
- **The exploration behind these choices** is in [notebooks/01_airfrans_exploration.ipynb](notebooks/01_airfrans_exploration.ipynb). It covers what was measured, which alternatives were rejected and why, for example k-nearest-neighbour graphs versus the real mesh edges.

### Modelling and evaluation

- **Inputs:** 7 per mesh node: position, signed distance to the airfoil, inflow velocity and surface normal.
- **Outputs:** u_x, u_y, p and ν_t. Each is scaled by the free stream's own scale (U∞, ½U∞², U∞·c), so the model learns how the flow depends on shape rather than on flow speed.
- **One shared evaluator** scores every model at full mesh resolution. It reads the test set once per model and refuses a second test evaluation, so test results can't steer modelling choices.
- **Deterministic training:** the same seed gives a bit-for-bit identical model. Every run writes `data/models/<run_id>/metrics.json` with its parameters, seed, dataset fingerprint and code commit.
- **265 automated tests,** including synthetic meshes with known answers. They run inside the container on the GPU.

## Running it

Everything runs in Docker, so no local Python or CUDA installation is needed. You need:
- Docker Desktop (WSL2 backend on Windows);
- an NVIDIA GPU with a current driver;
- about 35 GB of free disk for the raw, extracted and processed data.

The PyTorch wheels inside the image bring their own CUDA runtime.

```bash
cd servicios
cp .env.example .env    # replace CHANGE_ME with a generated token
docker compose --profile ml build
docker compose --profile ml run --rm ml       # test suite, on the GPU

# The pipeline, stage by stage. Ingestion and ETL are idempotent.
docker compose --profile ml run --rm ml python -m src.ingest.airfrans     # download, verify, extract
docker compose --profile ml run --rm ml python -m src.etl.pipeline        # VTK → parquet, splits, normalisation
docker compose --profile ml run --rm ml python -m src.train.mlp --seed 0  # add --hidden 64 for the smaller MLP
docker compose --profile ml run --rm ml python -m src.evaluate baseline thin_airfoil --split test
docker compose --profile ml run --rm ml python -m src.evaluate run <run_id> --split test
docker compose --profile ml run --rm ml python -m src.evaluate report

docker compose --profile dev up -d            # JupyterLab at http://localhost:8889
```

`ml` is a job runner, not a server. Data is written to `data/` through a bind mount and is never part of an image.

## Limitations

- **Scope:** 2D, incompressible and subsonic flow over NACA 4- and 5-digit airfoils, at Reynolds numbers of 2–6 million. That is the range AirfRANS covers.
- **Lift is pressure-only,** without the wall-shear contribution, and drag is not computed.
- **Each airfoil has ground truth at one angle of attack only.** A predicted lift curve can therefore be checked against the dataset at a single point per airfoil.

## Data

[AirfRANS](https://github.com/Extrality/airfrans_lib) (Bonnet et al., NeurIPS 2022 Datasets and Benchmarks Track) contains incompressible RANS solutions over NACA 4- and 5-digit airfoils, at Reynolds numbers of 2–6 million, in the subsonic regime. The dataset is licensed under [ODbL-1.0](https://opendatacommons.org/licenses/odbl/1-0/). It is **not redistributed here**; the pipeline downloads it from the authors' official host.

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
