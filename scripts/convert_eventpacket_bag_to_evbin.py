#!/usr/bin/env python3
"""Convert JetPilot EventPacket messages in a ROS 2 bag to EVSBIN v1.

Run this helper with the ROS/OpenEB Python interpreter that provides
``event_camera_py``. It intentionally has no dependency on RVT or PyTorch.
"""

import argparse
import json
import os
import struct
from pathlib import Path
from typing import List, Optional

import numpy as np
from rosbags.highlevel import AnyReader


HEADER = struct.Struct("<8sIIIIQQ24s")
MAGIC = b"EVSBENCH"
EVENT_DTYPE = np.dtype([
    ("t", "<i8"),
    ("x", "<u2"),
    ("y", "<u2"),
    ("p", "u1"),
    ("padding", "u1", (3,)),
])


class EvbinWriter:
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("wb")
        self.stream.write(HEADER.pack(MAGIC, 1, HEADER.size, 0, 0, 0, 1000, bytes(24)))
        self.width = 0
        self.height = 0
        self.event_count = 0

    def append(self, events: np.ndarray, width: int, height: int) -> None:
        if not len(events):
            return
        if self.width == 0:
            self.width, self.height = width, height
        elif (width, height) != (self.width, self.height):
            raise ValueError(
                f"EventPacket geometry changed from {self.width}x{self.height} to {width}x{height}"
            )
        self.stream.write(events.astype(EVENT_DTYPE, copy=False).tobytes())
        self.event_count += len(events)

    def close(self) -> None:
        if self.stream.closed:
            return
        self.stream.seek(0)
        self.stream.write(HEADER.pack(
            MAGIC, 1, HEADER.size, self.width, self.height,
            self.event_count, 1000, bytes(24),
        ))
        self.stream.close()


class BoundedChunkReorder:
    """Vectorized bounded reordering for high-rate decoded event chunks."""

    def __init__(self, reorder_window_us: int, flush_interval_us: int):
        self.reorder_window_us = reorder_window_us
        self.flush_interval_us = flush_interval_us
        self.pending: List[np.ndarray] = []
        self.max_seen_us: Optional[int] = None
        self.last_compaction_us: Optional[int] = None
        self.last_emitted_us: Optional[int] = None
        self.late_events = 0
        self.dropped_events = 0

    def add(self, events: np.ndarray) -> Optional[np.ndarray]:
        if not len(events):
            return None
        self.pending.append(events)
        chunk_max = int(events["t"].max())
        self.max_seen_us = chunk_max if self.max_seen_us is None else max(self.max_seen_us, chunk_max)
        if self.last_compaction_us is None:
            self.last_compaction_us = self.max_seen_us
            return None
        if self.max_seen_us - self.last_compaction_us < self.flush_interval_us:
            return None
        self.last_compaction_us = self.max_seen_us
        return self._flush(final=False)

    def finish(self) -> Optional[np.ndarray]:
        return self._flush(final=True)

    def _flush(self, final: bool) -> Optional[np.ndarray]:
        if not self.pending:
            return None
        events = np.concatenate(self.pending)
        order = np.argsort(events["t"], kind="stable")
        events = events[order]
        if self.last_emitted_us is not None:
            late = events["t"] < self.last_emitted_us
            self.late_events += int(late.sum())
            self.dropped_events += int(late.sum())
            events = events[~late]
        if not len(events):
            self.pending = []
            return None

        if final:
            emit_count = len(events)
        else:
            watermark = int(self.max_seen_us) - self.reorder_window_us
            emit_count = int(np.searchsorted(events["t"], watermark, side="right"))
        emitted = events[:emit_count]
        remaining = events[emit_count:]
        self.pending = [remaining] if len(remaining) else []
        if len(emitted):
            self.last_emitted_us = int(emitted["t"][-1])
            return emitted
        return None


def decoded_to_events(decoded: np.ndarray) -> np.ndarray:
    events = np.zeros(len(decoded), dtype=EVENT_DTYPE)
    if len(decoded):
        events["t"] = decoded["t"]
        events["x"] = decoded["x"]
        events["y"] = decoded["y"]
        events["p"] = np.asarray(decoded["p"], dtype=np.uint8)
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topic", default="/event_camera/events")
    parser.add_argument("--reorder-window-us", type=int, default=10_000)
    parser.add_argument("--flush-interval-us", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.reorder_window_us < 0 or args.flush_interval_us <= 0:
        raise ValueError("reorder-window-us must be non-negative and flush-interval-us must be positive")
    if not args.bag.exists():
        raise FileNotFoundError(args.bag)
    try:
        import event_camera_py
    except ImportError as error:
        raise RuntimeError(
            "event_camera_py is required. Run this converter from JetPilot's ROS/OpenEB environment."
        ) from error

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".in_progress")
    writer = EvbinWriter(temporary)
    reorder = BoundedChunkReorder(args.reorder_window_us, args.flush_interval_us)
    decoder = event_camera_py.Decoder()
    packet_count = 0
    decoded_count = 0
    width = 0
    height = 0
    try:
        with AnyReader([args.bag]) as reader:
            connections = [connection for connection in reader.connections if connection.topic == args.topic]
            if not connections:
                available = sorted({connection.topic for connection in reader.connections})
                raise ValueError(f"Topic {args.topic!r} not found. Available topics: {available}")
            for connection, _, rawdata in reader.messages(connections=connections):
                message = reader.deserialize(rawdata, connection.msgtype)
                width, height = int(message.width), int(message.height)
                decoder.decode_bytes(
                    str(message.encoding), width, height,
                    int(message.time_base), bytes(message.events),
                )
                decoded = decoder.get_cd_events()
                packet_count += 1
                if decoded is None or len(decoded) == 0:
                    continue
                decoded_count += len(decoded)
                emitted = reorder.add(decoded_to_events(decoded))
                if emitted is not None:
                    writer.append(emitted, width, height)
        emitted = reorder.finish()
        if emitted is not None:
            writer.append(emitted, width, height)
        writer.close()
        if writer.event_count == 0:
            raise RuntimeError(f"No CD events were decoded from topic {args.topic}")
        os.replace(temporary, args.output)
    except Exception:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise

    metadata = {
        "source_bag": str(args.bag),
        "topic": args.topic,
        "width": writer.width,
        "height": writer.height,
        "packet_count": packet_count,
        "decoded_events": decoded_count,
        "written_events": writer.event_count,
        "late_events": reorder.late_events,
        "dropped_events": reorder.dropped_events,
        "reorder_window_us": args.reorder_window_us,
        "flush_interval_us": args.flush_interval_us,
    }
    args.output.with_suffix(args.output.suffix + ".conversion.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
