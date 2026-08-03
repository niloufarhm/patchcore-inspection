import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms.functional as TF


class PatchCoreMemoryInspector:
    """
    Interpret PatchCore memory-bank embeddings.

    Main capabilities
    -----------------
    1. Decode a memory-bank index into:
         training image, feature-grid row, feature-grid column

    2. Display:
         - source image
         - approximate PatchMaker 3x3 footprint
         - local context crop

    3. Retrieve nearest memory-bank embeddings and display their
       corresponding source regions.

    4. Test embedding sensitivity to:
         - blur
         - grayscale
         - brightness
         - contrast

    5. Compute an occlusion-sensitivity map to estimate which input
       regions influence the selected patch embedding.
    """

    def __init__(
        self,
        model,
        train_dataset,
        selected_indices,
        device=None,
        memory_bank=None,
        patchsize=3,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.selected_indices = np.asarray(
            selected_indices,
            dtype=np.int64,
        )

        self.device = (
            device
            if device is not None
            else getattr(model, "device", torch.device("cpu"))
        )

        self.patchsize = patchsize

        # Infer the final PatchCore embedding grid.
        sample = self.train_dataset[0]
        sample_image = sample["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings, patch_shapes = self.model._embed(
                sample_image,
                detach=True,
                provide_patch_shapes=True,
            )

        self.patch_shapes = patch_shapes
        self.grid_h, self.grid_w = patch_shapes[0]
        self.patches_per_image = self.grid_h * self.grid_w

        # Try to obtain the already-fitted memory bank automatically.
        if memory_bank is None:
            memory_bank = self._find_memory_bank()

        self.memory_bank = np.asarray(memory_bank)

        if len(self.memory_bank) != len(self.selected_indices):
            raise ValueError(
                "Memory-bank size and selected-index count do not match:\n"
                f"memory bank: {len(self.memory_bank)}\n"
                f"indices:     {len(self.selected_indices)}"
            )

        print(
            f"Inspector ready\n"
            f"Embedding grid: {self.grid_h} × {self.grid_w}\n"
            f"Patches/image:  {self.patches_per_image}\n"
            f"Memory vectors: {len(self.memory_bank)}\n"
            f"Embedding dim:  {self.memory_bank.shape[1]}"
        )

    def _find_memory_bank(self):
        """
        Locate the fitted PatchCore memory bank.

        The standard PatchCore implementation stores it in:
            model.anomaly_scorer.detection_features[0]
        """
        scorer = getattr(self.model, "anomaly_scorer", None)

        if scorer is not None:
            detection_features = getattr(
                scorer,
                "detection_features",
                None,
            )

            if detection_features is not None:
                if isinstance(detection_features, (list, tuple)):
                    return detection_features[0]

                return detection_features

        raise AttributeError(
            "Could not automatically find the memory bank.\n"
            "Pass it explicitly using:\n\n"
            "PatchCoreMemoryInspector(..., memory_bank=your_memory_bank)"
        )

    # ------------------------------------------------------------------
    # Image conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _imagenet_mean(device="cpu"):
        return torch.tensor(
            [0.485, 0.456, 0.406],
            device=device,
        )[:, None, None]

    @staticmethod
    def _imagenet_std(device="cpu"):
        return torch.tensor(
            [0.229, 0.224, 0.225],
            device=device,
        )[:, None, None]

    def unnormalize(self, image_tensor):
        image = image_tensor.detach().cpu().float()

        mean = self._imagenet_mean()
        std = self._imagenet_std()

        image = image * std + mean
        image = image.clamp(0, 1)

        return image

    def normalize(self, image_tensor):
        image = image_tensor.float()

        mean = self._imagenet_mean(image.device)
        std = self._imagenet_std(image.device)

        return (image - mean) / std

    def display_image(self, image_tensor):
        image = self.unnormalize(image_tensor)
        return image.permute(1, 2, 0).numpy()

    # ------------------------------------------------------------------
    # Index decoding
    # ------------------------------------------------------------------

    def decode(self, memory_id):
        """
        Convert a memory-bank index to source-image and grid coordinates.
        """
        memory_id = int(memory_id)

        if not 0 <= memory_id < len(self.selected_indices):
            raise IndexError(
                f"memory_id must be between 0 and "
                f"{len(self.selected_indices) - 1}"
            )

        original_index = int(
            self.selected_indices[memory_id]
        )

        image_index = (
            original_index // self.patches_per_image
        )

        patch_index = (
            original_index % self.patches_per_image
        )

        row = patch_index // self.grid_w
        col = patch_index % self.grid_w

        if image_index >= len(self.train_dataset):
            raise IndexError(
                f"Decoded image index {image_index} exceeds dataset "
                f"length {len(self.train_dataset)}.\n"
                "This usually means the fitting DataLoader order differed "
                "from the dataset order."
            )

        return {
            "memory_id": memory_id,
            "original_index": original_index,
            "image_index": image_index,
            "patch_index": patch_index,
            "row": row,
            "col": col,
        }

    # ------------------------------------------------------------------
    # Spatial regions
    # ------------------------------------------------------------------

    def _grid_center(self, image_h, image_w, row, col):
        cell_h = image_h / self.grid_h
        cell_w = image_w / self.grid_w

        center_y = (row + 0.5) * cell_h
        center_x = (col + 0.5) * cell_w

        return center_x, center_y, cell_w, cell_h

    def _region_box(
        self,
        image_h,
        image_w,
        row,
        col,
        cells,
    ):
        """
        Convert a feature-grid neighborhood into an input-image box.

        This shows the spatial PatchMaker footprint, not the complete
        CNN theoretical receptive field.
        """
        center_x, center_y, cell_w, cell_h = (
            self._grid_center(
                image_h,
                image_w,
                row,
                col,
            )
        )

        region_w = cells * cell_w
        region_h = cells * cell_h

        x1 = max(0, int(center_x - region_w / 2))
        x2 = min(image_w, int(center_x + region_w / 2))
        y1 = max(0, int(center_y - region_h / 2))
        y2 = min(image_h, int(center_y + region_h / 2))

        return x1, y1, x2, y2

    def get_source_crop(
        self,
        memory_id,
        context_cells=7,
    ):
        info = self.decode(memory_id)

        sample = self.train_dataset[info["image_index"]]
        image_tensor = sample["image"]
        image = self.display_image(image_tensor)

        image_h, image_w = image.shape[:2]

        box = self._region_box(
            image_h=image_h,
            image_w=image_w,
            row=info["row"],
            col=info["col"],
            cells=context_cells,
        )

        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        return crop, image, box, info

    # ------------------------------------------------------------------
    # Nearest memory vectors
    # ------------------------------------------------------------------

    def nearest_memory_vectors(
        self,
        memory_id,
        k=8,
        exclude_same_image=False,
    ):
        query = self.memory_bank[memory_id]

        distances = np.linalg.norm(
            self.memory_bank - query[None, :],
            axis=1,
        )

        order = np.argsort(distances)

        selected_ids = []
        selected_distances = []

        query_info = self.decode(memory_id)

        for candidate_id in order:
            candidate_id = int(candidate_id)

            if candidate_id == memory_id:
                continue

            if exclude_same_image:
                candidate_info = self.decode(candidate_id)

                if (
                    candidate_info["image_index"]
                    == query_info["image_index"]
                ):
                    continue

            selected_ids.append(candidate_id)
            selected_distances.append(
                float(distances[candidate_id])
            )

            if len(selected_ids) == k:
                break

        return (
            np.asarray(selected_ids),
            np.asarray(selected_distances),
        )

    # ------------------------------------------------------------------
    # Embedding extraction at one location
    # ------------------------------------------------------------------

    def embedding_at(
        self,
        normalized_image,
        row,
        col,
    ):
        """
        Extract the final PatchCore embedding at one aligned grid location.

        normalized_image may be:
            [C,H,W] or [B,C,H,W]
        """
        if normalized_image.ndim == 3:
            normalized_image = normalized_image.unsqueeze(0)

        normalized_image = normalized_image.to(self.device)

        with torch.no_grad():
            embeddings, patch_shapes = self.model._embed(
                normalized_image,
                detach=True,
                provide_patch_shapes=True,
            )

        embeddings = np.asarray(embeddings)

        batch_size = normalized_image.shape[0]
        expected = (
            batch_size * self.grid_h * self.grid_w
        )

        if len(embeddings) != expected:
            raise ValueError(
                f"Expected {expected} embeddings, "
                f"received {len(embeddings)}."
            )

        embeddings = embeddings.reshape(
            batch_size,
            self.grid_h * self.grid_w,
            -1,
        )

        flat_patch_index = row * self.grid_w + col

        return embeddings[:, flat_patch_index, :]

    # ------------------------------------------------------------------
    # Perturbation tests
    # ------------------------------------------------------------------

    def _perturb(self, normalized_image, name):
        """
        Apply a controlled transformation to the full source image.
        """
        raw = self.unnormalize(
            normalized_image
        ).to(self.device)

        if name == "blur":
            changed = TF.gaussian_blur(
                raw,
                kernel_size=[11, 11],
                sigma=[3.0, 3.0],
            )

        elif name == "grayscale":
            changed = TF.rgb_to_grayscale(
                raw,
                num_output_channels=3,
            )

        elif name == "brightness":
            changed = TF.adjust_brightness(
                raw,
                brightness_factor=1.3,
            ).clamp(0, 1)

        elif name == "contrast":
            changed = TF.adjust_contrast(
                raw,
                contrast_factor=1.5,
            ).clamp(0, 1)

        else:
            raise ValueError(
                f"Unknown perturbation: {name}"
            )

        return self.normalize(changed)

    def perturbation_report(self, memory_id):
        info = self.decode(memory_id)

        sample = self.train_dataset[info["image_index"]]
        image = sample["image"].to(self.device)

        original_vector = self.embedding_at(
            normalized_image=image,
            row=info["row"],
            col=info["col"],
        )[0]

        transformations = [
            "blur",
            "grayscale",
            "brightness",
            "contrast",
        ]

        report = {}

        for name in transformations:
            changed_image = self._perturb(
                image,
                name,
            )

            changed_vector = self.embedding_at(
                normalized_image=changed_image,
                row=info["row"],
                col=info["col"],
            )[0]

            l2_change = np.linalg.norm(
                original_vector - changed_vector
            )

            cosine_similarity = np.dot(
                original_vector,
                changed_vector,
            ) / (
                np.linalg.norm(original_vector)
                * np.linalg.norm(changed_vector)
                + 1e-12
            )

            relative_change = (
                l2_change
                / (
                    np.linalg.norm(original_vector)
                    + 1e-12
                )
            )

            report[name] = {
                "l2_change": float(l2_change),
                "relative_change": float(relative_change),
                "cosine_similarity": float(
                    cosine_similarity
                ),
            }

        return report

    # ------------------------------------------------------------------
    # Occlusion sensitivity
    # ------------------------------------------------------------------

    def occlusion_sensitivity(
        self,
        memory_id,
        tiles=(14, 14),
        batch_size=16,
    ):
        """
        Empirically identify which image regions influence the selected
        embedding.

        Each tile is occluded with the normalized ImageNet mean, which is
        zero in normalized space. The embedding change is measured at the
        same PatchCore grid location.

        Returns
        -------
        sensitivity : [tile_rows, tile_cols] NumPy array
        """
        info = self.decode(memory_id)

        sample = self.train_dataset[info["image_index"]]
        image = sample["image"].to(self.device)

        original_vector = self.embedding_at(
            normalized_image=image,
            row=info["row"],
            col=info["col"],
        )[0]

        channels, image_h, image_w = image.shape
        tile_rows, tile_cols = tiles

        y_edges = np.linspace(
            0,
            image_h,
            tile_rows + 1,
            dtype=int,
        )

        x_edges = np.linspace(
            0,
            image_w,
            tile_cols + 1,
            dtype=int,
        )

        occluded_images = []
        tile_locations = []

        for tile_row in range(tile_rows):
            for tile_col in range(tile_cols):
                changed = image.clone()

                y1 = y_edges[tile_row]
                y2 = y_edges[tile_row + 1]
                x1 = x_edges[tile_col]
                x2 = x_edges[tile_col + 1]

                # Zero corresponds approximately to ImageNet mean
                # in normalized input space.
                changed[:, y1:y2, x1:x2] = 0

                occluded_images.append(changed)
                tile_locations.append(
                    (tile_row, tile_col)
                )

        sensitivity = np.zeros(
            (tile_rows, tile_cols),
            dtype=np.float32,
        )

        for start in range(
            0,
            len(occluded_images),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(occluded_images),
            )

            image_batch = torch.stack(
                occluded_images[start:end]
            )

            changed_vectors = self.embedding_at(
                normalized_image=image_batch,
                row=info["row"],
                col=info["col"],
            )

            changes = np.linalg.norm(
                changed_vectors
                - original_vector[None, :],
                axis=1,
            )

            for local_index, change in enumerate(changes):
                tile_row, tile_col = tile_locations[
                    start + local_index
                ]

                sensitivity[tile_row, tile_col] = change

        return sensitivity

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def show_source(
        self,
        memory_id,
        context_cells=7,
    ):
        crop, image, context_box, info = (
            self.get_source_crop(
                memory_id,
                context_cells=context_cells,
            )
        )

        image_h, image_w = image.shape[:2]

        patchmaker_box = self._region_box(
            image_h=image_h,
            image_w=image_w,
            row=info["row"],
            col=info["col"],
            cells=self.patchsize,
        )

        center_x, center_y, _, _ = (
            self._grid_center(
                image_h,
                image_w,
                info["row"],
                info["col"],
            )
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 5),
        )

        axes[0].imshow(image)

        # Larger box used for readable visual context.
        x1, y1, x2, y2 = context_box
        axes[0].add_patch(
            mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                linewidth=2,
                linestyle="--",
                label="Context crop",
            )
        )

        # Approximate spatial footprint of the 3x3 PatchMaker window.
        px1, py1, px2, py2 = patchmaker_box
        axes[0].add_patch(
            mpatches.Rectangle(
                (px1, py1),
                px2 - px1,
                py2 - py1,
                fill=False,
                linewidth=2,
                label=f"PatchMaker {self.patchsize}×{self.patchsize}",
            )
        )

        axes[0].scatter(
            center_x,
            center_y,
            marker="x",
            s=80,
        )

        axes[0].set_title(
            f"Memory vector {memory_id}\n"
            f"image={info['image_index']}, "
            f"grid=({info['row']}, {info['col']})"
        )

        axes[0].legend()
        axes[0].axis("off")

        axes[1].imshow(crop)
        axes[1].set_title(
            "Visual context region"
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        return info

    def show_neighbours(
        self,
        memory_id,
        k=8,
        context_cells=7,
        exclude_same_image=True,
    ):
        neighbour_ids, distances = (
            self.nearest_memory_vectors(
                memory_id=memory_id,
                k=k,
                exclude_same_image=exclude_same_image,
            )
        )

        display_ids = [memory_id] + list(neighbour_ids)
        display_distances = [None] + list(distances)

        columns = 3
        rows = math.ceil(
            len(display_ids) / columns
        )

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4 * columns, 4 * rows),
        )

        axes = np.asarray(axes).reshape(-1)

        for display_index, (
            axis,
            current_id,
            distance,
        ) in enumerate(
            zip(
                axes,
                display_ids,
                display_distances,
            )
        ):
            crop, _, _, info = self.get_source_crop(
                current_id,
                context_cells=context_cells,
            )

            axis.imshow(crop)

            if display_index == 0:
                title = (
                    f"Query: memory {current_id}\n"
                    f"image {info['image_index']}, "
                    f"grid ({info['row']},{info['col']})"
                )
            else:
                title = (
                    f"NN {display_index}: memory {current_id}\n"
                    f"d={distance:.3f}, "
                    f"image {info['image_index']}\n"
                    f"grid ({info['row']},{info['col']})"
                )

            axis.set_title(title)
            axis.axis("off")

        for axis in axes[len(display_ids):]:
            axis.axis("off")

        plt.tight_layout()
        plt.show()

        return neighbour_ids, distances

    def show_perturbation_report(
        self,
        memory_id,
    ):
        report = self.perturbation_report(
            memory_id
        )

        names = list(report.keys())

        relative_changes = [
            report[name]["relative_change"]
            for name in names
        ]

        cosine_values = [
            report[name]["cosine_similarity"]
            for name in names
        ]

        fig, axis = plt.subplots(
            figsize=(8, 4)
        )

        x = np.arange(len(names))
        width = 0.38

        axis.bar(
            x - width / 2,
            relative_changes,
            width,
            label="Relative L2 change",
        )

        axis.bar(
            x + width / 2,
            1 - np.asarray(cosine_values),
            width,
            label="1 - cosine similarity",
        )

        axis.set_xticks(x)
        axis.set_xticklabels(names)
        axis.set_ylabel("Embedding change")
        axis.set_title(
            f"Feature sensitivity: memory vector {memory_id}"
        )
        axis.legend()

        plt.tight_layout()
        plt.show()

        return report

    def show_occlusion_map(
        self,
        memory_id,
        tiles=(14, 14),
        batch_size=16,
        alpha=0.5,
    ):
        info = self.decode(memory_id)

        sample = self.train_dataset[info["image_index"]]
        image = self.display_image(
            sample["image"]
        )

        sensitivity = self.occlusion_sensitivity(
            memory_id=memory_id,
            tiles=tiles,
            batch_size=batch_size,
        )

        # Matplotlib stretches the coarse map to image size.
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 5),
        )

        axes[0].imshow(image)
        axes[0].set_title(
            f"Source image {info['image_index']}"
        )
        axes[0].axis("off")

        axes[1].imshow(image)
        heatmap = axes[1].imshow(
            sensitivity,
            interpolation="bilinear",
            extent=(0, image.shape[1], image.shape[0], 0),
            alpha=alpha,
        )

        axes[1].set_title(
            "Occlusion sensitivity\n"
            "Brighter = stronger influence on embedding"
        )
        axes[1].axis("off")

        fig.colorbar(
            heatmap,
            ax=axes[1],
            fraction=0.046,
            pad=0.04,
            label="Embedding change",
        )

        plt.tight_layout()
        plt.show()

        return sensitivity

    # ------------------------------------------------------------------
    # Complete inspection
    # ------------------------------------------------------------------

    def inspect(
        self,
        memory_id,
        k=8,
        context_cells=7,
        exclude_same_image=True,
        run_perturbations=True,
        run_occlusion=False,
        occlusion_tiles=(14, 14),
    ):
        print("=" * 70)
        print(f"Inspecting memory vector {memory_id}")
        print("=" * 70)

        info = self.show_source(
            memory_id=memory_id,
            context_cells=context_cells,
        )

        neighbour_ids, distances = (
            self.show_neighbours(
                memory_id=memory_id,
                k=k,
                context_cells=context_cells,
                exclude_same_image=exclude_same_image,
            )
        )

        report = None

        if run_perturbations:
            report = self.show_perturbation_report(
                memory_id
            )

            print("\nPerturbation report")
            print("-" * 70)

            for name, values in report.items():
                print(
                    f"{name:12s} | "
                    f"relative change: "
                    f"{values['relative_change']:.4f} | "
                    f"cosine: "
                    f"{values['cosine_similarity']:.4f}"
                )

        sensitivity = None

        if run_occlusion:
            sensitivity = self.show_occlusion_map(
                memory_id=memory_id,
                tiles=occlusion_tiles,
            )

        return {
            "source": info,
            "neighbour_ids": neighbour_ids,
            "neighbour_distances": distances,
            "perturbations": report,
            "occlusion_sensitivity": sensitivity,
        }