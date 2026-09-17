"""JetPilot EVSBIN v1 input and RVT-compatible event representations."""

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np
import torch


EVBIN_HEADER = struct.Struct("<8sIIIIQQ24s")
EVBIN_MAGIC = b"EVSBENCH"
EVBIN_EVENT_DTYPE = np.dtype([
    ("t", "<i8"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("p", "u1"),
    ("padding", "u1", (3,)),
])


@dataclass(frozen=True)
class EvbinInfo:
    path: Path
    width: int
    height: int
    event_count: int
    timestamp_unit_ns: int
    header_bytes: int


@dataclass(frozen=True)
class EventFrame:
    index: int
    timestamp_us: int
    relative_timestamp_us: int
    event_count: int
    representation: torch.Tensor


def read_evbin_info(path: Path) -> EvbinInfo:
    path = Path(path)
    with path.open("rb") as stream:
        raw_header = stream.read(EVBIN_HEADER.size)
    if len(raw_header) != EVBIN_HEADER.size:
        raise ValueError(f"Truncated EVSBIN header: {path}")
    magic, version, header_bytes, width, height, event_count, timestamp_unit_ns, _ = \
        EVBIN_HEADER.unpack(raw_header)
    if magic != EVBIN_MAGIC or version != 1 or header_bytes != EVBIN_HEADER.size:
        raise ValueError(f"Invalid or unsupported EVSBIN v1 header: {path}")
    if timestamp_unit_ns != 1000:
        raise ValueError(f"EVSBIN timestamps must use microseconds, got {timestamp_unit_ns} ns")
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid EVSBIN geometry: {width}x{height}")
    expected_size = header_bytes + event_count * EVBIN_EVENT_DTYPE.itemsize
    actual_size = os.stat(path).st_size
    if actual_size < expected_size:
        raise ValueError(
            f"Truncated EVSBIN payload: expected at least {expected_size} bytes, found {actual_size}"
        )
    return EvbinInfo(
        path=path,
        width=int(width),
        height=int(height),
        event_count=int(event_count),
        timestamp_unit_ns=int(timestamp_unit_ns),
        header_bytes=int(header_bytes),
    )


def open_evbin(path: Path) -> Tuple[EvbinInfo, np.ndarray]:
    info = read_evbin_info(path)
    if info.event_count == 0:
        return info, np.empty(0, dtype=EVBIN_EVENT_DTYPE)
    events = np.memmap(str(info.path), dtype=EVBIN_EVENT_DTYPE, mode="r",
                       offset=info.header_bytes, shape=(info.event_count,))
    validation_chunk = 5_000_000
    previous_last = None
    for start in range(0, info.event_count, validation_chunk):
        chunk = np.asarray(events["t"][start:start + validation_chunk])
        if previous_last is not None and len(chunk) and int(chunk[0]) < previous_last:
            raise ValueError("EVSBIN timestamps are not monotonic; reconvert with the reorder policy")
        if len(chunk) > 1 and bool(np.any(chunk[1:] < chunk[:-1])):
            raise ValueError("EVSBIN timestamps are not monotonic; reconvert with the reorder policy")
        if len(chunk):
            previous_last = int(chunk[-1])
    return info, events


def evbin_lower_bound(events: np.ndarray, target_us: int, right: bool = False) -> int:
    """Binary search timestamps without materializing the strided timestamp field."""
    left = 0
    end = len(events)
    while left < end:
        middle = left + (end - left) // 2
        timestamp = int(events[middle]["t"])
        move_right = timestamp <= target_us if right else timestamp < target_us
        if move_right:
            left = middle + 1
        else:
            end = middle
    return left


def _transform_coordinates(
        x: np.ndarray,
        y: np.ndarray,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    valid = (x < source_width) & (y < source_height)

    if mode == "stretch":
        target_x = x.astype(np.int64) * target_width // source_width
        target_y = y.astype(np.int64) * target_height // source_height
    elif mode == "center_crop":
        source_aspect = source_width / source_height
        target_aspect = target_width / target_height
        if source_aspect > target_aspect:
            crop_width = max(1, round(source_height * target_aspect))
            offset_x = (source_width - crop_width) // 2
            valid &= (x >= offset_x) & (x < offset_x + crop_width)
            target_x = (x.astype(np.int64) - offset_x) * target_width // crop_width
            target_y = y.astype(np.int64) * target_height // source_height
        else:
            crop_height = max(1, round(source_width / target_aspect))
            offset_y = (source_height - crop_height) // 2
            valid &= (y >= offset_y) & (y < offset_y + crop_height)
            target_x = x.astype(np.int64) * target_width // source_width
            target_y = (y.astype(np.int64) - offset_y) * target_height // crop_height
    elif mode == "letterbox":
        scale = min(target_width / source_width, target_height / source_height)
        scaled_width = max(1, round(source_width * scale))
        scaled_height = max(1, round(source_height * scale))
        offset_x = (target_width - scaled_width) // 2
        offset_y = (target_height - scaled_height) // 2
        target_x = x.astype(np.int64) * scaled_width // source_width + offset_x
        target_y = y.astype(np.int64) * scaled_height // source_height + offset_y
    else:
        raise ValueError(f"geometry_mode must be center_crop, letterbox, or stretch; got {mode!r}")

    valid &= (target_x >= 0) & (target_x < target_width)
    valid &= (target_y >= 0) & (target_y < target_height)
    return target_x[valid], target_y[valid], valid


def make_rvt_stacked_histogram(
        events: np.ndarray,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        bins: int = 10,
        count_cutoff: int = 10,
        geometry_mode: str = "center_crop") -> torch.Tensor:
    """Match RVT's uint8 StackedHistogram channel and time-bin contract."""
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")
    if not 1 <= count_cutoff <= 255:
        raise ValueError(f"count_cutoff must be within [1, 255], got {count_cutoff}")
    representation = np.zeros((2, bins, target_height, target_width), dtype=np.uint32)
    if len(events) == 0:
        return torch.from_numpy(representation.reshape(2 * bins, target_height, target_width).astype(np.uint8))

    target_x, target_y, valid = _transform_coordinates(
        events["x"], events["y"],
        source_size=(source_width, source_height),
        target_size=(target_width, target_height),
        mode=geometry_mode,
    )
    if not bool(valid.any()):
        return torch.from_numpy(representation.reshape(2 * bins, target_height, target_width).astype(np.uint8))

    timestamps = events["t"][valid].astype(np.int64)
    polarity = np.clip(events["p"][valid].astype(np.int64), 0, 1)
    time_span = max(int(timestamps[-1]) - int(timestamps[0]), 1)
    time_bins = np.minimum(
        bins - 1,
        (timestamps - int(timestamps[0])) * bins // time_span,
    ).astype(np.int64)
    np.add.at(representation, (polarity, time_bins, target_y, target_x), 1)
    np.minimum(representation, count_cutoff, out=representation)
    return torch.from_numpy(representation.reshape(2 * bins, target_height, target_width).astype(np.uint8))


def iter_rvt_event_frames(
        path: Path,
        target_height: int,
        target_width: int,
        bins: int = 10,
        window_ms: float = 50.0,
        stride_ms: float = 50.0,
        count_cutoff: int = 10,
        geometry_mode: str = "center_crop",
        start_offset_ms: float = 0.0,
        duration_ms: Optional[float] = None) -> Iterator[EventFrame]:
    info, events = open_evbin(path)
    if info.event_count == 0:
        raise ValueError(f"EVSBIN contains no events: {path}")
    if window_ms <= 0 or stride_ms <= 0:
        raise ValueError("window_ms and stride_ms must be positive")
    if start_offset_ms < 0 or (duration_ms is not None and duration_ms <= 0):
        raise ValueError("start_offset_ms must be non-negative and duration_ms must be positive or null")

    window_us = max(1, round(window_ms * 1000))
    stride_us = max(1, round(stride_ms * 1000))
    recording_start_us = int(events["t"][0])
    recording_end_us = int(events["t"][-1])
    segment_start_us = recording_start_us + round(start_offset_ms * 1000)
    segment_end_us = recording_end_us + 1
    if duration_ms is not None:
        segment_end_us = min(segment_end_us, segment_start_us + round(duration_ms * 1000))

    first_boundary_us = segment_start_us + window_us
    if first_boundary_us > segment_end_us:
        return
    number_of_frames = 1 + (segment_end_us - first_boundary_us) // stride_us
    for frame_index in range(number_of_frames):
        end_us = first_boundary_us + frame_index * stride_us
        start_us = end_us - window_us
        begin_index = evbin_lower_bound(events, start_us)
        end_index = evbin_lower_bound(events, end_us, right=True)
        window = events[begin_index:end_index]
        representation = make_rvt_stacked_histogram(
            events=window,
            source_width=info.width,
            source_height=info.height,
            target_width=target_width,
            target_height=target_height,
            bins=bins,
            count_cutoff=count_cutoff,
            geometry_mode=geometry_mode,
        )
        yield EventFrame(
            index=frame_index,
            timestamp_us=end_us,
            relative_timestamp_us=end_us - segment_start_us,
            event_count=len(window),
            representation=representation,
        )
