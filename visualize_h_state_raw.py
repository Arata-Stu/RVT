"""Run RVT detection and hidden-state visualization from JetPilot EVS recordings."""

import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Set

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from data.utils.raw_events import iter_rvt_event_frames, read_evbin_info
from data.utils.spatial import get_dataloading_hw
from data.utils.types import LstmStates
from models.detection.yolox.utils.boxes import postprocess
from modules.utils.fetch import fetch_model_module
from visualize_h_state import (
    StageScaleTracker,
    _activation_to_bgr,
    _add_export_title,
    _compose_frame,
    _event_representation_to_bgr,
    _make_panel,
    _prepare_image_export,
    _reduce_hidden_state,
    _safe_filename_component,
    _validate_stages,
    _write_export_image,
)


def _source_type(path: Path, configured: str) -> str:
    configured = configured.lower()
    if configured != "auto":
        if configured not in {"raw", "evbin", "rosbag"}:
            raise ValueError(f"source.type must be auto, raw, evbin, or rosbag; got {configured!r}")
        return configured
    suffix = path.suffix.lower()
    if suffix == ".raw":
        return "raw"
    if suffix == ".evbin":
        return "evbin"
    if path.is_dir() or suffix in {".mcap", ".db3"}:
        return "rosbag"
    raise ValueError(f"Could not infer source type from {path}; set source.type explicitly")


def _cache_path(source_path: Path, output_path: Path, configured: Any) -> Path:
    if configured is not None:
        return Path(hydra.utils.to_absolute_path(str(configured)))
    source_name = source_path.stem if source_path.is_file() else source_path.name
    return output_path.parent / f"{_safe_filename_component(source_name)}.rvt.evbin"


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    resolved = shutil.which(value)
    if resolved is None:
        raise FileNotFoundError(
            f"Executable {value!r} was not found. Build JetPilot tools/evs_benchmark "
            "or set source.raw_converter to evs_raw_to_evbin."
        )
    return resolved


def _prepare_evbin(config: DictConfig, output_path: Path) -> Path:
    source_path = Path(hydra.utils.to_absolute_path(str(config.source.path)))
    if not source_path.exists():
        raise FileNotFoundError(f"EVS source not found: {source_path}")
    kind = _source_type(source_path, str(config.source.type))
    if kind == "evbin":
        read_evbin_info(source_path)
        return source_path

    cache_path = _cache_path(source_path, output_path, config.source.evbin_cache)
    if cache_path.exists() and bool(config.source.reuse_cache):
        read_evbin_info(cache_path)
        print(f"Using cached EVSBIN: {cache_path}")
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if kind == "raw":
        converter = _resolve_executable(str(config.source.raw_converter))
        command = [
            converter,
            str(source_path),
            str(cache_path),
            "--timestamp-policy", "reorder",
            "--reorder-window-us", str(int(config.source.reorder_window_us)),
        ]
    else:
        converter = Path(__file__).parent / "scripts" / "convert_eventpacket_bag_to_evbin.py"
        command = [
            str(config.source.rosbag_python),
            str(converter),
            "--bag", str(source_path),
            "--output", str(cache_path),
            "--topic", str(config.source.event_topic),
            "--reorder-window-us", str(int(config.source.reorder_window_us)),
        ]
    print("Converting source to JetPilot EVSBIN v1...")
    print(" ".join(command))
    subprocess.run(command, check=True)
    read_evbin_info(cache_path)
    return cache_path


def _draw_detections(
        event_image: np.ndarray,
        detections: Optional[torch.Tensor],
        dataset_name: str) -> np.ndarray:
    image = event_image.copy()
    label_map = ("car", "pedestrian") if dataset_name == "gen1" else (
        "pedestrian", "two wheeler", "car"
    )
    colors = ((0, 255, 255), (255, 255, 0), (0, 128, 255))
    if detections is None:
        return image
    height, width = image.shape[:2]
    for detection in detections.detach().float().cpu().numpy():
        x1, y1, x2, y2, object_confidence, class_confidence, class_id = detection
        x1 = int(np.clip(round(x1), 0, width - 1))
        y1 = int(np.clip(round(y1), 0, height - 1))
        x2 = int(np.clip(round(x2), 0, width - 1))
        y2 = int(np.clip(round(y2), 0, height - 1))
        class_index = int(class_id) % len(label_map)
        confidence = float(object_confidence * class_confidence)
        color = colors[class_index % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            image, f"{label_map[class_index]} {confidence:.2f}",
            (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return image


def _serialize_detections(detections: Optional[torch.Tensor]) -> Any:
    if detections is None:
        return []
    output = []
    for row in detections.detach().float().cpu().numpy():
        output.append({
            "xyxy": [float(value) for value in row[:4]],
            "object_confidence": float(row[4]),
            "class_confidence": float(row[5]),
            "confidence": float(row[4] * row[5]),
            "class_id": int(row[6]),
        })
    return output


@hydra.main(config_path="config", config_name="visualize_h_state_raw", version_base="1.2")
def main(config: DictConfig) -> None:
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    output_path = Path(hydra.utils.to_absolute_path(str(config.visualization.output)))
    evbin_path = _prepare_evbin(config, output_path)
    evbin_info = read_evbin_info(evbin_path)
    source_path = Path(hydra.utils.to_absolute_path(str(config.source.path)))
    sequence_name = source_path.stem if source_path.is_file() else source_path.name
    sequence_stem = _safe_filename_component(sequence_name)

    target_height, target_width = get_dataloading_hw(config.dataset)
    bins = int(config.representation.bins)
    if bins * 2 != int(config.model.backbone.input_channels):
        raise ValueError(
            f"representation produces {bins * 2} channels but the model expects "
            f"{config.model.backbone.input_channels}"
        )

    device = torch.device(str(config.visualization.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; set visualization.device=cpu if intended")
    checkpoint_path = Path(hydra.utils.to_absolute_path(str(config.checkpoint)))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(checkpoint_path), full_config=config, map_location="cpu")
    module.eval().to(device)

    stages = _validate_stages(config.visualization.stages, module.mdl.backbone.num_stages)
    reduction = str(config.visualization.channel_reduction)
    tracker = StageScaleTracker(
        percentile=float(config.visualization.percentile),
        decay=float(config.visualization.scale_ema_decay),
    )
    write_video = bool(config.visualization.write_video)
    (export_enabled, export_frames, export_dir, image_format,
     jpeg_quality, export_titles) = _prepare_image_export(config)
    if not write_video and not export_enabled:
        raise ValueError("Enable visualization.write_video or visualization.image_export.enable")

    max_frames = config.visualization.max_frames
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("visualization.max_frames must be positive or null")
    panel_size = (int(config.visualization.panel_width), int(config.visualization.panel_height))
    if min(panel_size) <= 36 or panel_size[0] % 2 or panel_size[1] % 2:
        raise ValueError(f"Panel dimensions must be even and greater than 36, got {panel_size}")
    fps = float(config.visualization.fps)
    if fps <= 0:
        raise ValueError("visualization.fps must be positive")

    detections_enabled = bool(config.detections.enable)
    confidence_threshold = config.detections.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = config.model.postprocess.confidence_threshold
    confidence_threshold = float(confidence_threshold)

    writer = None
    panel_count = 1 + len(stages) + int(detections_enabled)
    if write_video:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_size = (panel_size[0] * 3, panel_size[1] * math.ceil(panel_count / 3) + 48)
        codec = str(config.visualization.codec)
        if len(codec) != 4:
            raise ValueError("visualization.codec must contain four characters")
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*codec), fps, output_size
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 writer: {output_path}")
    if export_enabled:
        export_dir.mkdir(parents=True, exist_ok=True)

    frames = iter_rvt_event_frames(
        path=evbin_path,
        target_height=target_height,
        target_width=target_width,
        bins=bins,
        window_ms=float(config.representation.window_ms),
        stride_ms=float(config.representation.stride_ms),
        count_cutoff=int(config.representation.count_cutoff),
        geometry_mode=str(config.source.geometry_mode),
        start_offset_ms=float(config.source.start_offset_ms),
        duration_ms=(None if config.source.duration_ms is None else float(config.source.duration_ms)),
    )
    metadata: Dict[str, Any] = {
        "source": str(source_path),
        "evbin": str(evbin_path),
        "source_geometry_wh": [evbin_info.width, evbin_info.height],
        "model_input_geometry_hw": [target_height, target_width],
        "geometry_mode": str(config.source.geometry_mode),
        "frame_numbering": "zero-based",
        "representation": OmegaConf.to_container(config.representation, resolve=True),
        "channel_order": (
            f"polarity_0_bins_0_to_{bins - 1}_then_polarity_1_bins_0_to_{bins - 1}"
        ),
        "detection_confidence_threshold": confidence_threshold,
        "frames": [],
    }
    previous_states: Optional[LstmStates] = None
    frames_processed = 0
    exported_frames: Set[int] = set()
    try:
        with torch.inference_mode():
            for frame in frames:
                if max_frames is not None and frames_processed >= int(max_frames):
                    break
                model_input = frame.representation.unsqueeze(0).to(device=device, dtype=module.dtype)
                model_input = module.input_padder.pad_tensor_ev_repr(model_input)
                backbone_features, states = module.mdl.forward_backbone(
                    x=model_input, previous_states=previous_states
                )
                previous_states = [(hidden.detach(), cell.detach()) for hidden, cell in states]

                processed_detections = None
                if detections_enabled:
                    predictions, _ = module.mdl.forward_detect(backbone_features)
                    processed_detections = postprocess(
                        predictions,
                        num_classes=int(config.model.head.num_classes),
                        conf_thre=confidence_threshold,
                        nms_thre=float(config.model.postprocess.nms_threshold),
                    )[0]

                event_image = _event_representation_to_bgr(frame.representation)
                detection_image = _draw_detections(
                    event_image, processed_detections, str(config.dataset.name)
                )
                panels = []
                if writer is not None:
                    panels.append(_make_panel(event_image, "Events", panel_size))
                    if detections_enabled:
                        panels.append(_make_panel(detection_image, "RVT detections", panel_size))

                stage_images: Dict[int, np.ndarray] = {}
                stage_metadata: Dict[str, Any] = {}
                for stage in stages:
                    hidden = states[stage - 1][0]
                    activation, signed = _reduce_hidden_state(hidden, reduction)
                    stride = module.mdl.backbone.get_strides((stage,))[0]
                    activation = activation[
                        :math.ceil(target_height / stride), :math.ceil(target_width / stride)
                    ]
                    scale = tracker.update(stage, activation, signed)
                    heatmap = _activation_to_bgr(activation, scale, signed)
                    stage_images[stage] = heatmap
                    stage_metadata[str(stage)] = {
                        "shape_hw": list(activation.shape), "scale": scale,
                    }
                    if writer is not None:
                        panels.append(_make_panel(
                            heatmap, f"h_state stage {stage} | {activation.shape} | scale {scale:.3g}", panel_size
                        ))

                if writer is not None:
                    header = (
                        f"source: {sequence_name} | frame: {frame.index:06d} | "
                        f"t: {frame.relative_timestamp_us / 1e6:.3f} s | events: {frame.event_count}"
                    )
                    writer.write(_compose_frame(panels, header=header))

                if export_enabled and frame.index in export_frames:
                    frame_stem = f"{sequence_stem}_frame_{frame.index:06d}"
                    exports = {"events": event_image}
                    if detections_enabled:
                        exports["detections"] = detection_image
                    filenames: Dict[str, str] = {}
                    for name, image in exports.items():
                        export_image = image
                        if export_titles:
                            export_image = _add_export_title(
                                image, f"source: {sequence_name} | frame: {frame.index:06d} | {name}"
                            )
                        filename = f"{frame_stem}_{name}.{image_format}"
                        _write_export_image(export_image, export_dir / filename, image_format, jpeg_quality)
                        filenames[name] = filename
                    for stage, image in stage_images.items():
                        export_image = image
                        if export_titles:
                            export_image = _add_export_title(
                                image, f"source: {sequence_name} | frame: {frame.index:06d} | h_state stage {stage}"
                            )
                        filename = f"{frame_stem}_h_state_stage_{stage}.{image_format}"
                        _write_export_image(export_image, export_dir / filename, image_format, jpeg_quality)
                        filenames[f"h_state_stage_{stage}"] = filename
                    metadata["frames"].append({
                        "frame": frame.index,
                        "timestamp_us": frame.timestamp_us,
                        "relative_timestamp_us": frame.relative_timestamp_us,
                        "event_count": frame.event_count,
                        "files": filenames,
                        "detections": _serialize_detections(processed_detections),
                        "stage_visualization": stage_metadata,
                    })
                    exported_frames.add(frame.index)

                frames_processed += 1
                if frames_processed % 100 == 0:
                    print(f"Processed {frames_processed} frames")
                if not write_video and export_enabled and exported_frames == export_frames:
                    break
    finally:
        if writer is not None:
            writer.release()

    if frames_processed == 0:
        if write_video:
            output_path.unlink(missing_ok=True)
        raise RuntimeError("No complete RVT event windows were produced from the selected source segment")
    if export_enabled:
        missing = sorted(export_frames - exported_frames)
        if missing:
            raise RuntimeError(f"Requested export frames were not reached: {missing}")
        metadata_path = export_dir / f"{sequence_stem}_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Exported {len(exported_frames)} frame(s) -> {export_dir}")
    if write_video:
        print(f"Wrote {frames_processed} frames -> {output_path}")
    print(
        "Note: JetPilot EVS differs from the Prophesee training domain; detections are valid model "
        "outputs but should be treated as out-of-domain until quantitatively validated."
    )


if __name__ == "__main__":
    main()
