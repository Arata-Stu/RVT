"""Render recurrent hidden states from a pretrained RVT checkpoint to MP4."""

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from data.genx_utils.labels import ObjectLabels
from data.genx_utils.sequence_for_streaming import SequenceForIter
from data.utils.types import DataType, DatasetType, LstmStates
from modules.utils.fetch import fetch_model_module


def _resolve_sequence(config: DictConfig) -> Tuple[Path, List[Path]]:
    split = str(config.visualization.split)
    if split not in {"train", "val", "test"}:
        raise ValueError(f"visualization.split must be train, val, or test, got {split!r}")

    split_path = Path(hydra.utils.to_absolute_path(str(config.dataset.path))) / split
    if not split_path.is_dir():
        raise FileNotFoundError(f"Dataset split directory not found: {split_path}")

    candidates = sorted(path for path in split_path.iterdir() if path.is_dir())
    if not candidates:
        raise RuntimeError(f"No sequence directories found in: {split_path}")

    sequence_name = config.visualization.sequence
    if sequence_name is not None:
        selected = split_path / str(sequence_name)
        if selected not in candidates:
            names = ", ".join(path.name for path in candidates[:10])
            suffix = " ..." if len(candidates) > 10 else ""
            raise FileNotFoundError(
                f"Sequence {sequence_name!r} not found in {split_path}. "
                f"Available examples: {names}{suffix}"
            )
        return selected, candidates

    sequence_index = int(config.visualization.sequence_index)
    if not -len(candidates) <= sequence_index < len(candidates):
        raise IndexError(
            f"sequence_index={sequence_index} is outside the available range "
            f"[-{len(candidates)}, {len(candidates) - 1}]"
        )
    return candidates[sequence_index], candidates


def _build_sequence(sequence_path: Path, config: DictConfig) -> SequenceForIter:
    dataset_name = str(config.dataset.name)
    if dataset_name == "gen1":
        dataset_type = DatasetType.GEN1
    elif dataset_name == "gen4":
        dataset_type = DatasetType.GEN4
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return SequenceForIter(
        path=sequence_path,
        ev_representation_name=str(config.dataset.ev_repr_name),
        sequence_length=int(config.dataset.sequence_length),
        dataset_type=dataset_type,
        downsample_by_factor_2=bool(config.dataset.downsample_by_factor_2),
    )


def _event_representation_to_bgr(event_repr: torch.Tensor) -> np.ndarray:
    array = event_repr.detach().float().cpu().numpy()
    channels, height, width = array.shape
    if channels <= 1 or channels % 2 != 0:
        raise ValueError(f"Expected an even number of event channels, got shape {array.shape}")

    half = channels // 2
    difference = array[half:].sum(axis=0) - array[:half].sum(axis=0)
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    image[difference > 0] = (255, 255, 255)
    image[difference < 0] = (0, 0, 0)
    return image


def _label_names(dataset_name: str) -> Tuple[str, ...]:
    if dataset_name == "gen1":
        return ("car", "pedestrian")
    if dataset_name == "gen4":
        # RVT's Gen4 preprocessing/evaluation keeps these three classes.
        return ("pedestrian", "two wheeler", "car")
    raise ValueError(f"Unsupported dataset for label visualization: {dataset_name}")


def _draw_ground_truth(
        event_image: np.ndarray,
        labels: Optional[ObjectLabels],
        dataset_name: str,
        line_thickness: int,
        font_scale: float) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    image = event_image.copy()
    if labels is None:
        return image, []

    names = _label_names(dataset_name)
    colors = (
        (0, 0, 255),      # pedestrian: red
        (255, 255, 0),    # two wheeler: cyan
        (0, 255, 255),    # car: yellow
        (255, 0, 255),
        (0, 165, 255),
        (255, 0, 0),
        (0, 255, 0),
    )
    height, width = image.shape[:2]
    values = labels.object_labels.detach().cpu().numpy()
    objects: List[Dict[str, Any]] = []
    for row in values:
        x, y, box_width, box_height = (float(value) for value in row[1:5])
        class_id = int(row[5])
        class_name = names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
        x0 = int(round(np.clip(x, 0, width - 1)))
        y0 = int(round(np.clip(y, 0, height - 1)))
        x1 = int(round(np.clip(x + box_width, 0, width - 1)))
        y1 = int(round(np.clip(y + box_height, 0, height - 1)))
        if x1 <= x0 or y1 <= y0:
            continue

        color = colors[class_id % len(colors)]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, line_thickness, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        text_top = max(0, y0 - text_height - baseline - 4)
        cv2.rectangle(
            image,
            (x0, text_top),
            (min(width - 1, x0 + text_width + 6), y0),
            color,
            -1,
        )
        cv2.putText(
            image,
            class_name,
            (x0 + 3, max(text_height, y0 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        objects.append({
            "class_id": class_id,
            "class_name": class_name,
            "bbox_xywh": [x, y, box_width, box_height],
        })
    return image, objects


def _reduce_hidden_state(hidden: torch.Tensor, reduction: str) -> Tuple[np.ndarray, bool]:
    if hidden.ndim != 4 or hidden.shape[0] != 1:
        raise ValueError(f"Expected hidden state shape [1, C, H, W], got {tuple(hidden.shape)}")

    hidden = hidden[0].detach().float()
    if reduction == "mean_abs":
        reduced = hidden.abs().mean(dim=0)
        signed = False
    elif reduction == "rms":
        reduced = hidden.square().mean(dim=0).sqrt()
        signed = False
    elif reduction == "max_abs":
        reduced = hidden.abs().amax(dim=0)
        signed = False
    elif reduction == "mean":
        reduced = hidden.mean(dim=0)
        signed = True
    else:
        raise ValueError(
            "visualization.channel_reduction must be one of: "
            f"mean_abs, rms, max_abs, mean; got {reduction!r}"
        )
    return reduced.cpu().numpy(), signed


class StageScaleTracker:
    """Track a robust activation scale independently for each recurrent stage."""

    def __init__(self, percentile: float, decay: float):
        if not 0 < percentile <= 100:
            raise ValueError(f"percentile must be in (0, 100], got {percentile}")
        if not 0 <= decay < 1:
            raise ValueError(f"scale_ema_decay must be in [0, 1), got {decay}")
        self.percentile = percentile
        self.decay = decay
        self.scales: Dict[int, float] = {}

    def update(self, stage: int, activation: np.ndarray, signed: bool) -> float:
        values = np.abs(activation) if signed else activation
        current = float(np.percentile(values, self.percentile))
        current = max(current, np.finfo(np.float32).eps)
        previous = self.scales.get(stage)
        # Expand immediately to avoid clipping sudden activity, but contract slowly
        # so that colors remain comparable across neighboring frames.
        smoothed = current if previous is None else self.decay * previous + (1 - self.decay) * current
        scale = max(current, smoothed)
        self.scales[stage] = scale
        return scale


def _activation_to_bgr(activation: np.ndarray, scale: float, signed: bool) -> np.ndarray:
    if signed:
        normalized = np.clip((activation / scale + 1.0) * 127.5, 0, 255)
        # TURBO gives negative/zero/positive activations distinct colors.
        return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)

    normalized = np.clip(activation / scale * 255.0, 0, 255)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)


def _make_panel(image: np.ndarray, title: str, panel_size: Tuple[int, int]) -> np.ndarray:
    panel_width, panel_height = panel_size
    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    available_height = panel_height - 36
    image_height, image_width = image.shape[:2]
    scale = min(panel_width / image_width, available_height / image_height)
    resized_width = max(1, round(image_width * scale))
    resized_height = max(1, round(image_height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
    x_offset = (panel_width - resized_width) // 2
    y_offset = 36 + (available_height - resized_height) // 2
    panel[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    cv2.putText(panel, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
    return panel


def _compose_frame(panels: Sequence[np.ndarray], header: str, columns: int = 3) -> np.ndarray:
    if not panels:
        raise ValueError("At least one panel is required")
    rows = math.ceil(len(panels) / columns)
    blank = np.zeros_like(panels[0])
    padded = list(panels) + [blank] * (rows * columns - len(panels))
    grid = np.concatenate(
        [np.concatenate(padded[row * columns:(row + 1) * columns], axis=1) for row in range(rows)],
        axis=0,
    )
    header_height = 48
    header_image = np.zeros((header_height, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        header_image, header, (14, 32), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.concatenate((header_image, grid), axis=0)


def _safe_filename_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return sanitized or "sequence"


def _add_export_title(image: np.ndarray, title: str) -> np.ndarray:
    title_height = 36
    output = np.zeros((image.shape[0] + title_height, image.shape[1], 3), dtype=np.uint8)
    output[title_height:] = image
    cv2.putText(output, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def _write_export_image(
        image: np.ndarray,
        path: Path,
        image_format: str,
        jpeg_quality: int) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if image_format == "jpg" else []
    if not cv2.imwrite(str(path), image, params):
        raise RuntimeError(f"Could not write image: {path}")


def _prepare_image_export(config: DictConfig) -> Tuple[bool, Set[int], Path, str, int, bool]:
    export_config = config.visualization.image_export
    enabled = bool(export_config.enable)
    frames = {int(frame) for frame in export_config.frames}
    if any(frame < 0 for frame in frames):
        raise ValueError(f"image_export.frames must contain non-negative frame numbers, got {sorted(frames)}")
    if enabled and not frames:
        raise ValueError("image_export.frames must not be empty when image_export.enable=true")

    image_format = str(export_config.format).lower().lstrip(".")
    if image_format == "jpeg":
        image_format = "jpg"
    if image_format not in {"png", "jpg"}:
        raise ValueError(f"image_export.format must be png or jpg, got {image_format!r}")

    jpeg_quality = int(export_config.jpeg_quality)
    if not 0 <= jpeg_quality <= 100:
        raise ValueError(f"image_export.jpeg_quality must be within [0, 100], got {jpeg_quality}")

    output_dir = Path(hydra.utils.to_absolute_path(str(export_config.output_dir)))
    return enabled, frames, output_dir, image_format, jpeg_quality, bool(export_config.include_titles)


def _iter_real_frames(
        sequence: SequenceForIter) -> Iterable[Tuple[torch.Tensor, Optional[ObjectLabels]]]:
    for sample_index in range(len(sequence)):
        sample = sequence[sample_index]
        event_representations = sample[DataType.EV_REPR]
        labels = sample[DataType.OBJLABELS_SEQ]
        padded_mask = sample[DataType.IS_PADDED_MASK]
        for event_repr, frame_labels, is_padded in zip(event_representations, labels, padded_mask):
            if not is_padded:
                yield event_repr, frame_labels


def _validate_stages(stages: Sequence[int], number_of_stages: int) -> List[int]:
    result = [int(stage) for stage in stages]
    if not result:
        raise ValueError("visualization.stages must contain at least one stage")
    if len(result) != len(set(result)):
        raise ValueError(f"visualization.stages contains duplicates: {result}")
    if min(result) < 1 or max(result) > number_of_stages:
        raise ValueError(f"visualization.stages must be within [1, {number_of_stages}], got {result}")
    return result


@hydra.main(config_path="config", config_name="visualize_h_state", version_base="1.2")
def main(config: DictConfig) -> None:
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    sequence_path, all_sequences = _resolve_sequence(config)
    sequence = _build_sequence(sequence_path=sequence_path, config=config)

    device = torch.device(str(config.visualization.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; set visualization.device=cpu if intended")

    checkpoint_path = Path(hydra.utils.to_absolute_path(str(config.checkpoint)))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(
        str(checkpoint_path), full_config=config, map_location="cpu"
    )
    module.eval().to(device)

    number_of_stages = module.mdl.backbone.num_stages
    stages = _validate_stages(config.visualization.stages, number_of_stages)
    reduction = str(config.visualization.channel_reduction)
    tracker = StageScaleTracker(
        percentile=float(config.visualization.percentile),
        decay=float(config.visualization.scale_ema_decay),
    )

    panel_size = (
        int(config.visualization.panel_width),
        int(config.visualization.panel_height),
    )
    if min(panel_size) <= 36:
        raise ValueError(f"panel dimensions must both be greater than 36, got {panel_size}")
    if panel_size[0] % 2 or panel_size[1] % 2:
        raise ValueError(f"panel dimensions must be even for MP4 encoding, got {panel_size}")

    fps = float(config.visualization.fps)
    if fps <= 0:
        raise ValueError(f"visualization.fps must be positive, got {fps}")
    max_frames: Optional[int] = config.visualization.max_frames
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError(f"visualization.max_frames must be positive or null, got {max_frames}")

    write_video = bool(config.visualization.write_video)
    (export_enabled, export_frames, export_dir, image_format,
     jpeg_quality, export_titles) = _prepare_image_export(config)
    if not write_video and not export_enabled:
        raise ValueError("Enable visualization.write_video or visualization.image_export.enable")

    output_path = Path(hydra.utils.to_absolute_path(str(config.visualization.output)))
    codec = str(config.visualization.codec)
    if len(codec) != 4:
        raise ValueError(f"visualization.codec must contain four characters, got {codec!r}")

    labels_enabled = bool(config.visualization.labels.enabled)
    label_line_thickness = int(config.visualization.labels.line_thickness)
    label_font_scale = float(config.visualization.labels.font_scale)
    if label_line_thickness <= 0:
        raise ValueError("visualization.labels.line_thickness must be positive")
    if label_font_scale <= 0:
        raise ValueError("visualization.labels.font_scale must be positive")

    number_of_panels = 1 + int(labels_enabled) + len(stages)
    output_size = (panel_size[0] * 3, panel_size[1] * math.ceil(number_of_panels / 3) + 48)
    writer: Optional[cv2.VideoWriter] = None
    if write_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            output_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer for {output_path} using codec {codec!r}")
    if export_enabled:
        export_dir.mkdir(parents=True, exist_ok=True)

    sequence_index = all_sequences.index(sequence_path)
    print(f"Sequence: {sequence_path.name} (index {sequence_index}, {sequence_index + 1}/{len(all_sequences)})")
    if write_video:
        print(f"Writing video: {output_path}")
    if export_enabled:
        print(f"Exporting frames {sorted(export_frames)} to: {export_dir}")
    previous_states: Optional[LstmStates] = None
    frames_written = 0
    exported_frames: Set[int] = set()
    sequence_stem = _safe_filename_component(sequence_path.name)
    export_metadata: Dict[str, Any] = {
        "sequence": sequence_path.name,
        "sequence_index": sequence_index,
        "split": str(config.visualization.split),
        "frame_numbering": "zero-based",
        "channel_reduction": reduction,
        "percentile": float(config.visualization.percentile),
        "scale_ema_decay": float(config.visualization.scale_ema_decay),
        "ground_truth_labels_enabled": labels_enabled,
        "frames": [],
    }
    try:
        with torch.inference_mode():
            for frame_index, (event_repr, frame_labels) in enumerate(_iter_real_frames(sequence)):
                if max_frames is not None and frames_written >= int(max_frames):
                    break

                model_input = event_repr.unsqueeze(0).to(device=device, dtype=module.dtype)
                model_input = module.input_padder.pad_tensor_ev_repr(model_input)
                _, states = module.mdl.forward_backbone(
                    x=model_input, previous_states=previous_states
                )
                previous_states = [(hidden.detach(), cell.detach()) for hidden, cell in states]

                event_image = _event_representation_to_bgr(event_repr)
                ground_truth_image: Optional[np.ndarray] = None
                ground_truth_objects: List[Dict[str, Any]] = []
                if labels_enabled:
                    ground_truth_image, ground_truth_objects = _draw_ground_truth(
                        event_image=event_image,
                        labels=frame_labels,
                        dataset_name=str(config.dataset.name),
                        line_thickness=label_line_thickness,
                        font_scale=label_font_scale,
                    )
                panels = []
                if writer is not None:
                    panels.append(_make_panel(
                        event_image,
                        f"Events | frame {frame_index}",
                        panel_size,
                    ))
                    annotation_status = (
                        f"{len(ground_truth_objects)} objects"
                        if frame_labels is not None else "no annotation"
                    )
                    if labels_enabled and ground_truth_image is not None:
                        panels.append(_make_panel(
                            ground_truth_image,
                            f"Ground truth | {annotation_status}",
                            panel_size,
                        ))
                stage_images: Dict[int, np.ndarray] = {}
                stage_metadata: Dict[str, Any] = {}
                for stage in stages:
                    hidden = states[stage - 1][0]
                    activation, signed = _reduce_hidden_state(hidden, reduction)
                    # Exclude the bottom/right area introduced by InputPadder.
                    stride = module.mdl.backbone.get_strides((stage,))[0]
                    valid_height = math.ceil(event_repr.shape[-2] / stride)
                    valid_width = math.ceil(event_repr.shape[-1] / stride)
                    activation = activation[:valid_height, :valid_width]
                    scale = tracker.update(stage, activation, signed)
                    heatmap = _activation_to_bgr(activation, scale, signed)
                    stage_images[stage] = heatmap
                    stage_metadata[str(stage)] = {
                        "shape_hw": list(activation.shape),
                        "scale": scale,
                    }
                    title = f"h_state stage {stage} | {activation.shape} | scale {scale:.3g}"
                    if writer is not None:
                        panels.append(_make_panel(heatmap, title, panel_size))

                if writer is not None:
                    header = (
                        f"sequence: {sequence_path.name} | sequence_index: {sequence_index} "
                        f"| frame: {frame_index:06d}"
                    )
                    writer.write(_compose_frame(panels, header=header))

                if export_enabled and frame_index in export_frames:
                    frame_stem = f"{sequence_stem}_frame_{frame_index:06d}"
                    event_export = event_image
                    if export_titles:
                        event_export = _add_export_title(
                            event_export, f"sequence: {sequence_path.name} | frame: {frame_index:06d} | events"
                        )
                    event_filename = f"{frame_stem}_events.{image_format}"
                    _write_export_image(event_export, export_dir / event_filename, image_format, jpeg_quality)

                    ground_truth_filename: Optional[str] = None
                    if labels_enabled and ground_truth_image is not None:
                        ground_truth_export = ground_truth_image
                        if export_titles:
                            ground_truth_export = _add_export_title(
                                ground_truth_export,
                                f"sequence: {sequence_path.name} | frame: {frame_index:06d} | ground truth",
                            )
                        ground_truth_filename = f"{frame_stem}_ground_truth.{image_format}"
                        _write_export_image(
                            ground_truth_export,
                            export_dir / ground_truth_filename,
                            image_format,
                            jpeg_quality,
                        )

                    stage_filenames: Dict[str, str] = {}
                    for stage, heatmap in stage_images.items():
                        stage_export = heatmap
                        if export_titles:
                            stage_export = _add_export_title(
                                stage_export,
                                f"sequence: {sequence_path.name} | frame: {frame_index:06d} | h_state stage {stage}",
                            )
                        stage_filename = f"{frame_stem}_h_state_stage_{stage}.{image_format}"
                        _write_export_image(stage_export, export_dir / stage_filename, image_format, jpeg_quality)
                        stage_filenames[str(stage)] = stage_filename

                    export_metadata["frames"].append({
                        "frame": frame_index,
                        "events": event_filename,
                        "ground_truth": ground_truth_filename,
                        "ground_truth_annotated": frame_labels is not None,
                        "ground_truth_objects": ground_truth_objects,
                        "stages": stage_filenames,
                        "stage_visualization": stage_metadata,
                    })
                    exported_frames.add(frame_index)

                frames_written += 1
                if frames_written % 100 == 0:
                    print(f"Rendered {frames_written} frames")
                if not write_video and export_enabled and exported_frames == export_frames:
                    break
    finally:
        if writer is not None:
            writer.release()

    if frames_written == 0:
        if write_video:
            output_path.unlink(missing_ok=True)
        raise RuntimeError(f"No non-padded frames were found in sequence {sequence_path.name}")
    if export_enabled:
        missing_frames = sorted(export_frames - exported_frames)
        if missing_frames:
            raise RuntimeError(
                f"Requested export frames were not reached: {missing_frames}. "
                "Check max_frames and the sequence length."
            )
        metadata_path = export_dir / f"{sequence_stem}_metadata.json"
        metadata_path.write_text(json.dumps(export_metadata, indent=2), encoding="utf-8")
        print(f"Exported {len(exported_frames)} frame(s) -> {export_dir}")
    if write_video:
        print(f"Done: {frames_written} frames -> {output_path}")


if __name__ == "__main__":
    main()
