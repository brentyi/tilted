# TILTED

**[Project page](https://brentyi.github.io/tilted/) &bull; [arXiv](https://arxiv.org/abs/2308.15461)**

Code release for our ICCV 2023 paper:

<table><tr><td>
    Brent Yi<sup>1</sup>, Weijia Zeng<sup>1</sup>, Sam Buchanan<sup>2</sup>, and Yi Ma<sup>1</sup>.
    <strong>Canonical Factors for Hybrid Neural Fields.</strong>
    International Conference on Computer Vision (ICCV), 2023.
</td></tr>
</table>
<sup>1</sup><em>UC Berkeley</em>, <sup>2</sup><em>TTI-Chicago</em>

## Overview

We study neural field architectures that rely on factored feature volumes, by
(1) analyzing factored grids in 2D to characterize undesirable biases for
axis-aligned signals, and (2) using the resulting insights to study TILTED, a
family of hybrid neural field architectures that removes these biases.

This repository is structured as follows:

```
.
├── tilted
│   ├── core                - Code shared between experiments. Factored grid
│   │                         and neural decoder implementations.
│   ├── rgb2d               - Neural radiance field rendering, training, and
│   │                         dataloading utilities.
│   ├── nerf                - 2D image reconstruction data and training
│   │                         utilities.
│   └── sdf                 - Signed distance field dataloading, training, and
│                             meshing infrastructure.
│
├── paper_commands          - Commands used for running paper experiments (NeRF)
├── paper_results           - Output files used to generate paper tables. (NeRF)
│                             Contains hyperparameters, evaluation metrics,
│                             runtimes, etc.
│
├── tables_nerf.ipynb       - Table generation notebook for NeRF experiments.
│
├── train_nerf.py           - Training script for neural radiance field experiments.
├── visualize_nerf.py       - Visualize trained neural radiance fields.
│
└── requirements.txt        - Python dependencies.
```

Note that training scripts for 2D and SDF experiments have not yet been
released. Feel free to reach out if you need these.

## Running

### Setup

This repository has been tested with Python 3.8, `jax==0.4.9`, and
`jaxlib==0.4.9+cuda11.cudnn86`. We recommend first installing JAX via their
official instructions: https://github.com/google/jax#installation

We've packaged dependencies into a `requirements.txt` file:

```
pip install -r requirements.txt
```

### Datasets

Meshes for SDF experiments were downloaded from
[alecjacobson/common-3d-test-models/](https://github.com/alecjacobson/common-3d-test-models/).

All NeRF datasets were downloaded using
[nerfstudio](https://github.com/nerfstudio-project/nerfstudio)'s
`ns-download-data` command:

```
# Requires nerfstudio installation.
ns-download-data blender
ns-download-data nerfstudio all
```

### Training

Commands we used for training NeRF models in the paper can be found in
`paper_commands/`.

Here are two examples, which should run at ~65 it/sec on an RTX 4090:

```
# Train a model on a synthetic scene.
python train_nerf.py blender-kplane-32c-axis-aligned --dataset-path {path_to_data}

# Train a model on a real scene.
python train_nerf.py nerfstudio-kplane-32c-axis-aligned --dataset-path {path_to_data}
```

The `--help` flag can also be passed in to print helptext.

### Visualization

Trained radiance fields can be interactively visualized. Helptext for the
visualization script can be found via:

```
python visualize_nerf.py --help
```

This supports RGB, PCA, and feature norm visualization:

https://github.com/brentyi/tilted/assets/6992947/f8fd1dff-0a78-4f91-9973-bbfb98c3af0c

The core viewer infrastructure has been moved into
[nerfstudio-project/viser](https://github.com/nerfstudio-project/viser), which
may be helpful if you're interested in visualization for other projects.

## Notes

This is research code, so parts of it may be chaotic. If you have questions or
comments, please reach out!

This material is based upon work supported by the National Science Foundation
Graduate Research Fellowship Program under Grant DGE 2146752. YM acknowledges
partial support from the ONR grant N00014-22-1-2102, the joint Simons
Foundation-NSF DMS grant 2031899, and a research grant from TBSI.

If any of it is useful, you can also cite:

```bibtex
@inproceedings{tilted2023,
    author = {Yi, Brent and Zeng, Weijia and Buchanan, Sam and Ma, Yi},
    title = {Canonical Factors for Hybrid Neural Fields},
    booktitle = {International Conference on Computer Vision (ICCV)},
    year = {2023},
}
```
