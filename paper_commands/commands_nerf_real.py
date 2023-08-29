"""Generate commands for running real-world experiments."""

nerfstudio_datasets = [
    "aspen",
    "bww_entrance",
    "campanile",
    "desolation",
    "dozer",
    "Egypt",
    "floating-tree",
    "Giannini-Hall",
    "kitchen",
    "library",
    "person",
    "plane",
    "poster",
    "redwoods2",
    "storefront",
    "sculpture",
    "stump",
    "vegetation",
]

for config in (
    "nerfstudio-kplane-64c-axis-aligned",
    "nerfstudio-kplane-64c-tilted-4t",
    "nerfstudio-kplane-64c-tilted-8t",
    "nerfstudio-vm-192c-axis-aligned",
    "nerfstudio-vm-192c-tilted-4t",
    "nerfstudio-vm-192c-tilted-8t",
):
    for dataset in nerfstudio_datasets:
        print(
            f"python train_nerf.py {config} --dataset-path ./data/nerfstudio/{dataset}"
        )
