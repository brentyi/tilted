"""Generate commands for running synthetic experiments."""

blender_datasets = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]

for scene_aligned in (True, False):
    for seed in (0, 1, 2):
        global_rotate_seed = None if scene_aligned else seed

        # Axis-aligned.
        for config in (
            "blender-kplane-64c-axis-aligned",
            "blender-vm-192c-axis-aligned",
        ):
            for dataset in blender_datasets:
                print(
                    f"python train_nerf.py {config}"
                    f" --dataset-path ./data/nerf_synthetic/{dataset}"
                    f" --seed {seed}"
                    f" --render-config.global-rotate-seed {global_rotate_seed}"
                )

        # TILTED.
        for enable_bottleneck in (True, False):
            for config in (
                "blender-kplane-64c-tilted-4t",
                "blender-kplane-64c-tilted-8t",
                "blender-vm-192c-tilted-4t",
                "blender-vm-192c-tilted-8t",
            ):
                for dataset in blender_datasets:
                    print(
                        f"python train_nerf.py {config}"
                        f" --dataset-path ./data/nerf_synthetic/{dataset}"
                        f" --seed {seed}"
                        f" --bottleneck.enable {enable_bottleneck}"
                        f" --render-config.global-rotate-seed {global_rotate_seed}"
                    )
