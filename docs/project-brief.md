# Project brief — airfoil-flow-surrogate

## Problem

A RANS simulation of the flow around one 2D airfoil takes minutes to hours of compute. Design iteration needs answers in seconds. A neural surrogate trained on high-fidelity simulations can predict the pressure and velocity fields for a new geometry and inflow condition in milliseconds. It is only useful if its errors are measured honestly on geometries it has never seen, in the quantities an aerodynamicist uses, such as surface pressure and lift.

## Goal

Build a reproducible pipeline that:

1. extracts and structures CFD result files (VTK unstructured meshes) into model-ready tables;
2. trains surrogates that map geometry and boundary conditions to flow fields: a per-point MLP and a graph neural network on the CFD mesh;
3. evaluates them against a naive baseline on unseen airfoils;
4. exports predictions back to VTK for side-by-side inspection in ParaView.

## Data

| | |
|---|---|
| Source | AirfRANS `Dataset.zip`, official host of the dataset authors |
| Format | per simulation: one `_internal.vtu` (volume mesh and fields), `_aerofoil.vtp` (surface), `_freestream.vtp` (far-field boundary) |
| Volume | 1,000 simulations; 10.0 GB compressed, 15.0 GB uncompressed |
| Cadence | static release (2022-08-11); downloaded once, snapshot recorded |
| Freshness | not applicable: a fixed benchmark |
| Splits | the dataset's own `manifest.json`: 800 training / 200 test simulations, every test airfoil unseen |
| License | ODbL-1.0, not redistributed |

## Success criteria

- Every derived dataset is reproducible from the raw snapshot by code in `src/`.
- Every model is reported next to the naive free-stream baseline on the same 200 test simulations, with the same evaluator.
- Reported metrics: per-variable relative L2 error on the fields, surface pressure-coefficient error, lift-coefficient error and rank correlation.
- Predictions open in ParaView on the original mesh.
- A fresh clone reproduces the results with the commands in the README.

## Out of scope

Drag prediction (requires wall shear stress), 3D geometries, meshing new geometries, and production deployment.
