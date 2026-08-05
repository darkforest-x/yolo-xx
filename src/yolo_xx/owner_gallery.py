"""Build the stratified, blind Owner review gallery for the 5m perfect pattern.

The gallery exists because there is currently no Owner-confirmed small-timeframe
box at all.  It turns authenticated 5m pre-holdout OHLCV into a small, auditable
set of rendered windows that the Owner can judge one by one.

Three separations are structural, not stylistic:

1. Candidate mining is not ground truth.  The broad spread filter and the legacy
   detectors only decide *what gets shown*; they never write a label.
2. The review UI is blind.  Bucket, model name, confidence, symbol, time, and
   any historical label live in ``review_manifest.json`` and never reach the
   HTML page.
3. Sources are authenticated before pandas parses a single CSV, and only assets
   the PR-00 registry marks as 5m / pre-holdout / DIRECT_REUSE or
   REVIEW_AND_REUSE may be read.

No training happens here, no historical asset is modified, and no outcome,
return, or backtest value is read.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd

from . import annotations as review_ledger
from .data import add_mas, load_ohlcv_csv
from .labels import DenseSegment, find_dense_segments, segment_to_bbox
from .pattern_spec import (
    GALLERY_BUCKETS,
    PatternSpecError,
    load_pattern_spec,
    pattern_spec_sha256,
    spec_summary,
)
from .render import IMG_HEIGHT, IMG_WIDTH, min_rel_span_for, render_chart
from .source_manifest import (
    HOLDOUT_START,
    load_source_manifest,
    sha256_file,
    utc_iso,
    verify_loaded_frame,
    verify_snapshot_identity,
)

GALLERY_SCHEMA_VERSION = 1
GALLERY_TYPE = "yolo_xx_owner_review_gallery"
DEFAULT_REGISTRY = "docs/asset_registry_v2.json"
DEFAULT_OUT = "reports/pr01a_owner_gallery"
DEFAULT_SEED = 20260805
ALLOWED_REUSE = ("DIRECT_REUSE", "REVIEW_AND_REUSE")
TIMEFRAME = "5m"
MA_PERIODS = (20, 60, 120)
WARMUP_BARS = 3 * max(MA_PERIODS)

# Distance between the end of a candidate pattern and the right edge of the
# rendered window.  Only the Owner-frozen right_context_bars are used.  Wider
# offsets were tried and dropped: with a fixed y-scale, later price action
# squashes an earlier compression into the corner, which makes the pattern hard
# to judge and would bias this review toward negative.  Position-shortcut
# auditing belongs to detection evaluation, not to pattern definition.
FROZEN_RIGHT_CONTEXT = (0, 8, 16, 24)
GALLERY_CONTEXT_OFFSETS = FROZEN_RIGHT_CONTEXT

RULE_BUCKETS = (
    "strong_rule_candidates",
    "longer_complete_candidates",
    "near_threshold_candidates",
    "fast_only_partial_dense",
)

LEGACY_PROPOSAL_WEIGHTS = (
    "runs/detect/hardneg_w96_v2_s/weights/best.pt",
    "runs/detect/owner_short_ab_w96_v2/weights/best.pt",
    "weights/baselines/owner_short_star_v10.pt",
)
LEGACY_CONF_FLOOR = 0.05
LEGACY_IMGSZ = 960
LEGACY_IOU = 0.7

# Perceptual near-duplicate rejection.  An 8x8 difference hash is useless on these
# charts: they are mostly white, so unrelated windows collapse to a distance of 1.
# A 16x16 hash (256 bits) separates them.  The threshold is anchored on a measured
# quantity rather than taste: re-rendering the same window shifted by one bar moves
# the hash by ~18 bits, so two gallery images that differ by less than that are, for
# review purposes, the same picture twice.
PHASH_SIZE = 16
PHASH_BITS = PHASH_SIZE * PHASH_SIZE
PHASH_MIN_DISTANCE = 18


class GalleryError(ValueError):
    """Raised when the gallery cannot be built from authenticated sources."""


# --------------------------------------------------------------------------- #
# registry gate
# --------------------------------------------------------------------------- #
def select_source_snapshots(registry_path: str | Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    """Return the PR-00 registry entries this gallery is allowed to read."""
    path = Path(registry_path)
    if not path.is_file():
        raise GalleryError(f"asset registry does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise GalleryError("asset registry has no assets list")
    selected = [
        asset
        for asset in assets
        if asset.get("asset_type") == "snapshot"
        and asset.get("timeframe") == TIMEFRAME
        and asset.get("holdout_status") == "preholdout"
        and asset.get("reuse_status") in ALLOWED_REUSE
        and asset.get("present_in_repository", False)
    ]
    if not selected:
        raise GalleryError(
            "no registry asset satisfies timeframe=5m, holdout_status=preholdout, "
            f"reuse_status in {ALLOWED_REUSE}"
        )
    return selected


# --------------------------------------------------------------------------- #
# candidate mining (never ground truth)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Candidate:
    """One mined window proposal, before any human judgement exists."""

    symbol: str
    bucket: str
    core_start: int
    core_end: int
    window_end: int
    raw_bars: int
    mean_full_spread: float
    min_full_spread: float
    mean_fast_spread: float
    slope_ratio: float
    context_offset: int
    score: float
    model_conf: float | None = None
    model_key: str | None = None


def find_runs(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Return inclusive index runs of a boolean mask, merging small gaps."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    runs: list[list[int]] = [[int(indices[0]), int(indices[0])]]
    for index in indices[1:]:
        value = int(index)
        if value - runs[-1][1] <= merge_gap + 1:
            runs[-1][1] = value
        else:
            runs.append([value, value])
    return [(start, end) for start, end in runs]


def _tightest_core(full_spread: np.ndarray, start: int, end: int, max_bars: int) -> tuple[int, int]:
    length = end - start + 1
    if length <= max_bars:
        return start, end
    best_index, best_score = start, float("inf")
    for index in range(start, end - max_bars + 2):
        values = full_spread[index : index + max_bars]
        if np.isnan(values).all():
            continue
        score = float(np.nanmean(values))
        if score < best_score:
            best_score, best_index = score, index
    return best_index, best_index + max_bars - 1


def _offset_for(symbol: str, core_end: int, seed: int, limit: int) -> int:
    """Pick a deterministic, pattern-specific distance to the right edge."""
    allowed = [value for value in GALLERY_CONTEXT_OFFSETS if value <= limit]
    if not allowed:
        return 0
    digest = hashlib.sha256(f"{seed}:{symbol}:{core_end}".encode("utf-8")).digest()
    return allowed[digest[0] % len(allowed)]


def mine_candidates(
    symbol: str,
    frame: pd.DataFrame,
    *,
    window_bars: int,
    fast_max: float,
    full_max: float,
    min_bars: int,
    max_bars: int,
    merge_gap: int,
    seed: int,
) -> list[Candidate]:
    """Mine rule, near-threshold, fast-only, and background proposals for one symbol.

    Every number here is a *screening* statistic.  None of them decides whether a
    window is a perfect pattern; only the Owner review does.
    """
    fast = pd.to_numeric(frame["fast_spread"], errors="coerce").to_numpy(dtype=float)
    full = pd.to_numeric(frame["full_spread"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    slow = pd.to_numeric(frame[f"sma{max(MA_PERIODS)}"], errors="coerce").to_numpy(dtype=float)
    total = len(frame)
    first_valid = WARMUP_BARS
    last_valid = total - 1

    strict = (fast <= fast_max) & (full <= full_max)
    relaxed = (fast <= fast_max * 1.35) & (full <= full_max * 1.35)
    fast_only = (fast <= fast_max) & (full > full_max)

    candidates: list[Candidate] = []

    def emit(bucket: str, start: int, end: int, raw_bars: int) -> None:
        core_start, core_end = _tightest_core(full, start, end, max_bars)
        core_length = core_end - core_start + 1
        limit = max(0, window_bars - core_length - 4)
        offset = _offset_for(symbol, core_end, seed, limit)
        window_end = core_end + offset
        if window_end > last_valid:
            window_end = last_valid
        window_start = window_end - window_bars + 1
        if window_start < first_valid or core_start < window_start:
            return
        core = slice(core_start, core_end + 1)
        mean_full = float(np.nanmean(full[core])) if not np.isnan(full[core]).all() else float("nan")
        min_full = float(np.nanmin(full[core])) if not np.isnan(full[core]).all() else float("nan")
        mean_fast = float(np.nanmean(fast[core])) if not np.isnan(fast[core]).all() else float("nan")
        price = float(close[core_end]) if np.isfinite(close[core_end]) else float("nan")
        if np.isfinite(slow[core_end]) and np.isfinite(slow[core_start]) and price:
            slope = float((slow[core_end] - slow[core_start]) / max(core_length, 1) / price)
        else:
            slope = float("nan")
        if not np.isfinite(mean_full):
            return
        candidates.append(
            Candidate(
                symbol=symbol,
                bucket=bucket,
                core_start=core_start,
                core_end=core_end,
                window_end=int(window_end),
                raw_bars=int(raw_bars),
                mean_full_spread=mean_full,
                min_full_spread=min_full,
                mean_fast_spread=mean_fast,
                slope_ratio=slope,
                context_offset=int(window_end - core_end),
                score=mean_full,
            )
        )

    for start, end in find_runs(strict, merge_gap):
        raw = end - start + 1
        if raw < min_bars:
            continue
        bucket = "longer_complete_candidates" if raw >= max_bars else "strong_rule_candidates"
        emit(bucket, start, end, raw)

    strict_runs = {(start, end) for start, end in find_runs(strict, merge_gap)}
    for start, end in find_runs(relaxed, merge_gap):
        raw = end - start + 1
        if raw < min_bars:
            continue
        overlaps_strict = any(not (end < s or start > e) for s, e in strict_runs)
        if overlaps_strict:
            continue
        emit("near_threshold_candidates", start, end, raw)

    for start, end in find_runs(fast_only, merge_gap):
        raw = end - start + 1
        if raw < min_bars:
            continue
        emit("fast_only_partial_dense", start, end, raw)

    return candidates


def background_endpoints(
    symbol: str,
    frame: pd.DataFrame,
    *,
    window_bars: int,
    fast_max: float,
    full_max: float,
    stride: int,
) -> list[Candidate]:
    """Return grid endpoints whose window contains no near-dense bar at all."""
    fast = pd.to_numeric(frame["fast_spread"], errors="coerce").to_numpy(dtype=float)
    full = pd.to_numeric(frame["full_spread"], errors="coerce").to_numpy(dtype=float)
    relaxed = (fast <= fast_max * 1.35) & (full <= full_max * 1.35)
    total = len(frame)
    out: list[Candidate] = []
    for window_end in range(WARMUP_BARS + window_bars - 1, total, stride):
        window = slice(window_end - window_bars + 1, window_end + 1)
        if relaxed[window].any():
            continue
        segment_full = full[window]
        if np.isnan(segment_full).all():
            continue
        out.append(
            Candidate(
                symbol=symbol,
                bucket="random_continuous_background",
                core_start=window_end - window_bars + 1,
                core_end=window_end,
                window_end=int(window_end),
                raw_bars=0,
                mean_full_spread=float(np.nanmean(segment_full)),
                min_full_spread=float(np.nanmin(segment_full)),
                mean_fast_spread=float(np.nanmean(fast[window])),
                slope_ratio=float("nan"),
                context_offset=0,
                score=float(np.nanmean(segment_full)),
            )
        )
    return out


def scan_grid_endpoints(frame: pd.DataFrame, *, window_bars: int, count: int) -> list[int]:
    """Return evenly spaced, non-overlapping window endpoints for a model scan."""
    first = WARMUP_BARS + window_bars - 1
    last = len(frame) - 1
    if last <= first:
        return []
    span = last - first
    step = max(window_bars, span // max(count, 1))
    return list(range(first, last + 1, step))[:count]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _time_bucket(open_time: pd.Timestamp) -> str:
    return f"{open_time.year}-{open_time.month:02d}"


def stratified_select(
    candidates: Sequence[Candidate],
    *,
    target: int,
    taken: dict[str, list[int]],
    window_bars: int,
    symbol_times: Mapping[str, pd.Series],
    seed: int,
    accept: "Callable[[Candidate], bool] | None" = None,
) -> list[Candidate]:
    """Pick ``target`` proposals spread over symbols and calendar months.

    Selection walks symbols round-robin, then months inside a symbol, so a single
    liquid symbol or a single volatile week cannot dominate a bucket.  Any window
    overlapping an already-selected window of the same symbol is skipped.

    ``accept`` is the perceptual gate.  It renders the candidate and rejects it if
    the picture is too close to one already in the gallery, so near-duplicate
    charts are never selected in the first place instead of being reported after
    the fact.
    """
    pools: dict[str, dict[str, list[Candidate]]] = {}
    for candidate in candidates:
        month = _time_bucket(symbol_times[candidate.symbol].iloc[candidate.window_end])
        pools.setdefault(candidate.symbol, {}).setdefault(month, []).append(candidate)
    for symbol, months in pools.items():
        for month, items in months.items():
            items.sort(key=lambda item: (item.score, item.window_end))

    rng = random.Random(seed)
    symbols = sorted(pools)
    rng.shuffle(symbols)
    selected: list[Candidate] = []
    # Start each symbol on a different month.  A shared cursor starting at zero
    # fills a bucket out of the earliest months and never reaches the most recent
    # ones, which is how the first build silently lost April entirely.
    month_cursor: dict[str, int] = {
        symbol: int(hashlib.sha256(f"{seed}:{symbol}".encode("utf-8")).hexdigest(), 16)
        % max(len(pools.get(symbol, {})), 1)
        for symbol in symbols
    }

    progressed = True
    while len(selected) < target and progressed:
        progressed = False
        for symbol in symbols:
            if len(selected) >= target:
                break
            months = sorted(pools.get(symbol, {}))
            if not months:
                continue
            for _ in range(len(months)):
                month = months[month_cursor[symbol] % len(months)]
                month_cursor[symbol] += 1
                queue = pools[symbol][month]
                while queue:
                    candidate = queue.pop(0)
                    used = taken.setdefault(candidate.symbol, [])
                    if any(
                        abs(candidate.window_end - other) < window_bars for other in used
                    ):
                        continue
                    if accept is not None and not accept(candidate):
                        continue
                    used.append(candidate.window_end)
                    selected.append(candidate)
                    progressed = True
                    break
                else:
                    continue
                break
    return selected


# --------------------------------------------------------------------------- #
# rendering and image identity
# --------------------------------------------------------------------------- #
def dhash(image: np.ndarray, *, size: int = PHASH_SIZE) -> str:
    """Return a difference hash used only for near-duplicate rejection."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def hamming(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def render_window(
    frame: pd.DataFrame,
    window_end: int,
    *,
    window_bars: int,
    out_path: Path | None,
) -> tuple[np.ndarray, Any]:
    """Render one 96-bar window with the frozen six-line 5m chart contract."""
    window = frame.iloc[window_end - window_bars + 1 : window_end + 1]
    return render_chart(
        window,
        width=IMG_WIDTH,
        height=IMG_HEIGHT,
        out_path=out_path,
        ma_periods=MA_PERIODS,
        min_rel_span=min_rel_span_for(TIMEFRAME),
    )


# --------------------------------------------------------------------------- #
# legacy model proposals (ranking only)
# --------------------------------------------------------------------------- #
def legacy_model_scores(
    image_dir: Path,
    weights: Sequence[Path],
    *,
    device: str,
    conf: float = LEGACY_CONF_FLOOR,
    batch: int = 16,
    cache: str | Path | None = None,
) -> dict[str, tuple[float, str]]:
    """Return the best legacy confidence per image path.

    The legacy detectors were trained on a different timeframe and a different
    line set, so their output is treated as an out-of-domain *ranking* signal and
    nothing else.  It never becomes a label.
    """
    directory = Path(image_dir).resolve()
    cache_path = Path(cache) if cache else None
    if cache_path is not None and cache_path.is_file():
        stored = json.loads(cache_path.read_text(encoding="utf-8"))
        if stored.get("weights") == [str(path) for path in weights] and stored.get("conf") == conf:
            restored = {
                str(directory / name): (float(value[0]), str(value[1]))
                for name, value in stored.get("scores", {}).items()
            }
            if restored:
                return restored

    from ultralytics import YOLO  # imported lazily: keeps unit tests model-free

    best: dict[str, tuple[float, str]] = {}
    for weight in weights:
        model = YOLO(str(weight))
        key = weight.parent.parent.name if weight.parent.name == "weights" else weight.stem
        # Directory source only.  A list source makes ultralytics load every image
        # into one batch and label results "image0.jpg", which loses the binding
        # between a prediction and the window it came from.
        for result in model.predict(
            source=str(directory),
            imgsz=LEGACY_IMGSZ,
            conf=conf,
            iou=LEGACY_IOU,
            batch=batch,
            device=device,
            save=False,
            save_txt=False,
            verbose=False,
            stream=True,
        ):
            raw_path = getattr(result, "path", None)
            if raw_path is None:
                raise GalleryError("legacy scan result is missing its source path")
            path = str(Path(raw_path).resolve())
            scores = result.boxes.conf.tolist() if result.boxes is not None else []
            if not scores:
                continue
            top = float(max(scores))
            current = best.get(path)
            if current is None or top > current[0]:
                best[path] = (top, key)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "weights": [str(path) for path in weights],
                    "conf": conf,
                    "scores": {Path(path).name: value for path, value in best.items()},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return best


# --------------------------------------------------------------------------- #
# gallery build
# --------------------------------------------------------------------------- #
@dataclass
class SymbolData:
    symbol: str
    frame: pd.DataFrame
    record: Any
    relative_path: str
    sha256: str
    open_times: pd.Series = field(init=False)

    def __post_init__(self) -> None:
        self.open_times = self.frame["open_time"]


def load_symbols(
    snapshots: Iterable[dict[str, Any]],
    *,
    repo: Path,
) -> list[SymbolData]:
    """Authenticate every declared source, then parse it into MA-annotated frames."""
    symbols: list[SymbolData] = []
    for asset in snapshots:
        manifest_path = repo / asset["path"] / "source_snapshot.json"
        snapshot = load_source_manifest(
            manifest_path,
            expected_timeframe=TIMEFRAME,
            end_before=HOLDOUT_START,
        )
        if snapshot.holdout_read:
            raise GalleryError(f"{manifest_path} declares holdout_read=true")
        verify_snapshot_identity(snapshot)
        for record in snapshot.files:
            frame = load_ohlcv_csv(
                record.path,
                timeframe=TIMEFRAME,
                strict_cadence=True,
            )
            verify_loaded_frame(frame, record, timeframe=TIMEFRAME)
            frame = add_mas(frame, periods=MA_PERIODS)
            symbol = record.relative_path.split("_USDT")[0].replace("okx_", "")
            symbols.append(
                SymbolData(
                    symbol=symbol,
                    frame=frame,
                    record=record,
                    relative_path=record.relative_path,
                    sha256=record.sha256,
                )
            )
    if not symbols:
        raise GalleryError("no authenticated 5m source file was loaded")
    return symbols


def _sample_id(symbol: str, window_end_time: pd.Timestamp, window_bars: int) -> str:
    stamp = window_end_time.strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol}_{TIMEFRAME}_w{window_bars}_{stamp}"


def build_owner_gallery(
    *,
    spec_path: str | Path = "configs/PERFECT_PATTERN_SPEC_V1.yaml",
    registry_path: str | Path = DEFAULT_REGISTRY,
    out_dir: str | Path = DEFAULT_OUT,
    repo: str | Path = ".",
    seed: int = DEFAULT_SEED,
    scan_pool_per_symbol: int = 120,
    legacy_weights: Sequence[str] = LEGACY_PROPOSAL_WEIGHTS,
    device: str = "mps",
    work_dir: str | Path | None = None,
    scores_cache: str | Path | None = None,
) -> dict[str, Any]:
    """Build the 240-image blind Owner gallery and its audit."""
    repo_root = Path(repo).resolve()
    spec = load_pattern_spec(spec_path)
    gallery_contract = spec["owner_gallery_contract"]
    window_bars = spec["window_contract"]["window_bars"]
    mining = spec["candidate_mining_contract"]["legacy_broad_filter"]
    total_images = gallery_contract["total_images"]
    per_bucket = gallery_contract["images_per_bucket"]
    buckets = list(gallery_contract["bucket_names"])

    assets = select_source_snapshots(registry_path)
    symbols = load_symbols(assets, repo=repo_root)
    symbol_times = {item.symbol: item.open_times for item in symbols}
    frames = {item.symbol: item.frame for item in symbols}
    sources = {item.symbol: item for item in symbols}

    pools: dict[str, list[Candidate]] = {name: [] for name in buckets}
    for item in symbols:
        mined = mine_candidates(
            item.symbol,
            item.frame,
            window_bars=window_bars,
            fast_max=float(mining["fast_spread_max"]),
            full_max=float(mining["full_spread_max"]),
            min_bars=int(mining["min_dense_bars"]),
            max_bars=int(mining["max_dense_bars"]),
            merge_gap=int(mining["merge_gap_bars"]),
            seed=seed,
        )
        for candidate in mined:
            pools[candidate.bucket].append(candidate)
        pools["random_continuous_background"].extend(
            background_endpoints(
                item.symbol,
                item.frame,
                window_bars=window_bars,
                fast_max=float(mining["fast_spread_max"]),
                full_max=float(mining["full_spread_max"]),
                stride=window_bars,
            )
        )

    # Bucket B should reward duration, not tightness alone.
    for candidate in pools["longer_complete_candidates"]:
        object.__setattr__(candidate, "score", -float(candidate.raw_bars))
    rng = random.Random(seed)
    rng.shuffle(pools["random_continuous_background"])
    for index, candidate in enumerate(pools["random_continuous_background"]):
        object.__setattr__(candidate, "score", float(index))

    work = Path(work_dir) if work_dir else Path(out_dir) / "_scan_pool"
    shutil.rmtree(work, ignore_errors=True)
    pool_dir = work / "pool"
    sel_dir = work / "selected"
    pool_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[tuple[str, int], dict[str, Any]] = {}
    accepted_hashes: list[tuple[str, str]] = []
    accepted_sha: dict[str, str] = {}
    rejected_duplicates: list[dict[str, Any]] = []

    def candidate_box(candidate: Candidate, transform: Any) -> list[float] | None:
        """Return the rule-mined box the Owner is asked to accept, adjust, or reject.

        Rule buckets carry their own segment.  For a model proposal or a background
        window the box, if any, comes from the same broad rule — never from the
        legacy model, whose prediction must not be shown as a pre-label.
        """
        window_end = candidate.window_end
        window_start = window_end - window_bars + 1
        window = frames[candidate.symbol].iloc[window_start : window_end + 1].reset_index(drop=True)
        if candidate.bucket in RULE_BUCKETS:
            segment = DenseSegment(candidate.core_start - window_start, candidate.core_end - window_start)
        else:
            found = find_dense_segments(
                window,
                fast_max=float(mining["fast_spread_max"]) * 1.35,
                full_max=float(mining["full_spread_max"]) * 1.35,
                min_bars=int(mining["min_dense_bars"]),
                merge_gap=int(mining["merge_gap_bars"]),
                max_bars=int(mining["max_dense_bars"]),
            )
            if not found:
                return None
            full = pd.to_numeric(window["full_spread"], errors="coerce").to_numpy(dtype=float)
            segment = min(
                found, key=lambda item: float(np.nanmean(full[item.start : item.end + 1]))
            )
        box = segment_to_bbox(window, segment, transform, ma_periods=MA_PERIODS)
        return [round(float(value), 6) for value in box] if box is not None else None

    def accept(candidate: Candidate) -> bool:
        """Render the candidate and refuse it if the Owner would see it twice."""
        key = (candidate.symbol, candidate.window_end)
        path = sel_dir / f"{candidate.symbol}_{candidate.window_end}.png"
        image, transform = render_window(
            frames[candidate.symbol], candidate.window_end, window_bars=window_bars, out_path=path
        )
        digest = sha256_file(path)
        phash = dhash(image)
        if digest in accepted_sha:
            rejected_duplicates.append(
                {
                    "kind": "image_sha256",
                    "symbol": candidate.symbol,
                    "window_end_index": candidate.window_end,
                    "duplicate_of": accepted_sha[digest],
                }
            )
            return False
        for other_key, other_hash in accepted_hashes:
            distance = hamming(phash, other_hash)
            if distance < PHASH_MIN_DISTANCE:
                rejected_duplicates.append(
                    {
                        "kind": "perceptual",
                        "symbol": candidate.symbol,
                        "window_end_index": candidate.window_end,
                        "duplicate_of": other_key,
                        "distance": distance,
                        "threshold": PHASH_MIN_DISTANCE,
                    }
                )
                return False
        identity = f"{candidate.symbol}:{candidate.window_end}"
        accepted_hashes.append((identity, phash))
        accepted_sha[digest] = identity
        rendered[key] = {
            "path": path,
            "sha256": digest,
            "phash": phash,
            "price_min": float(transform.price_min),
            "price_max": float(transform.price_max),
            "box": candidate_box(candidate, transform),
        }
        return True

    taken: dict[str, list[int]] = {}
    selected: dict[str, list[Candidate]] = {}
    for name in buckets:
        if name == "legacy_model_high_confidence_proposals":
            continue
        selected[name] = stratified_select(
            pools[name],
            target=per_bucket,
            taken=taken,
            window_bars=window_bars,
            symbol_times=symbol_times,
            seed=seed,
            accept=accept,
        )

    # ------------------------------------------------------------------ E --
    scan_index: dict[str, tuple[str, int]] = {}
    scan_paths: list[Path] = []
    for item in symbols:
        for window_end in scan_grid_endpoints(
            item.frame, window_bars=window_bars, count=scan_pool_per_symbol
        ):
            path = (pool_dir / f"{item.symbol}_{window_end}.png").resolve()
            render_window(item.frame, window_end, window_bars=window_bars, out_path=path)
            scan_index[str(path)] = (item.symbol, window_end)
            scan_paths.append(path)

    resolved_weights = [repo_root / weight for weight in legacy_weights]
    missing = [str(path) for path in resolved_weights if not path.is_file()]
    if missing:
        raise GalleryError(f"legacy proposal weights are missing: {', '.join(missing)}")
    scores = legacy_model_scores(pool_dir, resolved_weights, device=device, cache=scores_cache)

    proposals: list[Candidate] = []
    for path, (confidence, model_key) in scores.items():
        if path not in scan_index:
            raise GalleryError(f"legacy scan returned an unexpected image path: {path}")
        symbol, window_end = scan_index[path]
        frame = frames[symbol]
        window = slice(window_end - window_bars + 1, window_end + 1)
        full = pd.to_numeric(frame["full_spread"], errors="coerce").to_numpy(dtype=float)[window]
        fast = pd.to_numeric(frame["fast_spread"], errors="coerce").to_numpy(dtype=float)[window]
        if np.isnan(full).all():
            continue
        proposals.append(
            Candidate(
                symbol=symbol,
                bucket="legacy_model_high_confidence_proposals",
                core_start=window_end - window_bars + 1,
                core_end=window_end,
                window_end=window_end,
                raw_bars=0,
                mean_full_spread=float(np.nanmean(full)),
                min_full_spread=float(np.nanmin(full)),
                mean_fast_spread=float(np.nanmean(fast)),
                slope_ratio=float("nan"),
                context_offset=0,
                score=-confidence,
                model_conf=confidence,
                model_key=model_key,
            )
        )
    selected["legacy_model_high_confidence_proposals"] = stratified_select(
        proposals,
        target=per_bucket,
        taken=taken,
        window_bars=window_bars,
        symbol_times=symbol_times,
        seed=seed,
        accept=accept,
    )

    shortfalls = {
        name: per_bucket - len(items) for name, items in selected.items() if len(items) < per_bucket
    }
    if shortfalls:
        raise GalleryError(
            "not enough deduplicated candidates for bucket(s): "
            + ", ".join(f"{name} short {count}" for name, count in sorted(shortfalls.items()))
        )

    # ---------------------------------------------------------------- write --
    output = Path(out_dir)
    images_dir = output / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    ordered: list[tuple[str, Candidate]] = [
        (name, candidate) for name in buckets for candidate in selected[name]
    ]
    shuffle_rng = random.Random(seed + 1)
    shuffle_rng.shuffle(ordered)

    samples: list[dict[str, Any]] = []
    image_hashes: dict[str, str] = {}
    perceptual: list[tuple[str, str]] = []
    duplicate_events: list[dict[str, Any]] = []

    for index, (bucket, candidate) in enumerate(ordered, start=1):
        review_id = f"R{index:04d}"
        item = sources[candidate.symbol]
        frame = item.frame
        window_end = candidate.window_end
        window_start = window_end - window_bars + 1
        image_path = images_dir / f"{review_id}.png"
        cached = rendered[(candidate.symbol, window_end)]
        shutil.copyfile(cached["path"], image_path)
        digest = sha256_file(image_path)
        if digest != cached["sha256"]:
            raise GalleryError(f"{review_id}: copied image does not match the rendered digest")
        phash = cached["phash"]
        # Re-check the shipped set: the selection gate should already guarantee this,
        # and an assertion here is what proves it rather than assuming it.
        if digest in image_hashes:
            duplicate_events.append(
                {"kind": "image_sha256", "review_id": review_id, "duplicate_of": image_hashes[digest]}
            )
        image_hashes[digest] = review_id
        for other_id, other_hash in perceptual:
            distance = hamming(phash, other_hash)
            if distance < PHASH_MIN_DISTANCE:
                duplicate_events.append(
                    {
                        "kind": "perceptual",
                        "review_id": review_id,
                        "duplicate_of": other_id,
                        "distance": distance,
                    }
                )
        perceptual.append((review_id, phash))

        window_end_open = frame["open_time"].iloc[window_end]
        window_start_open = frame["open_time"].iloc[window_start]
        available_at = window_end_open + pd.Timedelta(minutes=5)
        sample_id = _sample_id(candidate.symbol, window_end_open, window_bars)
        core_left = max(candidate.core_start - window_start, 0)
        core_right = max(candidate.core_end - window_start, 0)
        samples.append(
            {
                "review_id": review_id,
                "sample_id": sample_id,
                "image": f"images/{review_id}.png",
                "image_sha256": digest,
                "perceptual_hash": phash,
                "bucket": bucket,
                "symbol": candidate.symbol,
                "timeframe": TIMEFRAME,
                "window_bars": window_bars,
                "window_start_index": int(window_start),
                "window_end_index": int(window_end),
                "window_start_open_time": utc_iso(window_start_open),
                "window_end_open_time": utc_iso(window_end_open),
                "available_at": utc_iso(available_at),
                "duplicate_group": f"{candidate.symbol}:{window_end // window_bars}",
                "source_file": item.relative_path,
                "source_sha256": item.sha256,
                "source_first_open_time": utc_iso(item.record.first_open_time),
                "source_last_open_time": utc_iso(item.record.last_open_time),
                "candidate_box": cached["box"],
                "candidate_box_origin": "rule_candidate" if cached["box"] else None,
                "candidate_box_is_ground_truth": False,
                "candidate_core_start_bar": int(core_left),
                "candidate_core_end_bar": int(core_right),
                "candidate_core_right_fraction": round((core_right + 1) / window_bars, 6),
                "candidate_raw_bars": int(candidate.raw_bars),
                "candidate_context_offset_bars": int(candidate.context_offset),
                "candidate_mean_full_spread": round(candidate.mean_full_spread, 8),
                "candidate_min_full_spread": round(candidate.min_full_spread, 8),
                "candidate_mean_fast_spread": round(candidate.mean_fast_spread, 8),
                "candidate_slope_ratio": (
                    None if not np.isfinite(candidate.slope_ratio) else round(candidate.slope_ratio, 10)
                ),
                "legacy_model_conf": candidate.model_conf,
                "legacy_model_key": candidate.model_key,
                "price_min": round(cached["price_min"], 10),
                "price_max": round(cached["price_max"], 10),
                "label_status": "unreviewed",
                "ground_truth": None,
            }
        )

    symbols_present = sorted({sample["symbol"] for sample in samples})
    times = sorted(sample["window_end_open_time"] for sample in samples)
    manifest = {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "manifest_type": GALLERY_TYPE,
        "task_id": spec["task_id"],
        "pattern_spec": spec_summary(spec),
        "pattern_spec_sha256": pattern_spec_sha256(spec),
        "spec_status": spec["status"],
        "created_from": "authenticated 5m pre-holdout OHLCV snapshots only",
        "registry": str(registry_path),
        "registry_assets": [asset["path"] for asset in assets],
        "seed": seed,
        "holdout_read": False,
        "outcome_used": False,
        "training_started": False,
        "ground_truth_source": "owner_review_only",
        "candidate_mining_contract": spec["candidate_mining_contract"],
        "position_policy": (
            "one image per candidate; the distance to the right edge is one of the "
            "Owner-frozen right_context_bars 0/8/16/24, chosen deterministically per "
            "candidate. Position diversity comes from real endpoints, not from "
            "re-showing the same pattern."
        ),
        "frozen_right_context_bars": list(FROZEN_RIGHT_CONTEXT),
        "legacy_scan_pool_images": len(scan_paths),
        "legacy_scan_conf_floor": LEGACY_CONF_FLOOR,
        "deduplication": {
            "source_endpoints": "one window per (symbol, endpoint); overlapping windows rejected",
            "image_sha256": "exact duplicates rejected during selection",
            "perceptual_hash_bits": PHASH_BITS,
            "perceptual_min_distance": PHASH_MIN_DISTANCE,
            "perceptual_threshold_anchor": "same window shifted by one bar moves the hash ~18 bits",
            "candidates_rejected_as_near_duplicates": len(rejected_duplicates),
        },
        "legacy_proposal_weights": [
            {
                "path": str(Path(weight).relative_to(repo_root)),
                "sha256": sha256_file(weight),
                "role": "candidate_ranking_only",
                "is_ground_truth": False,
            }
            for weight in resolved_weights
        ],
        "render": {
            "renderer": "yolo_xx.render.render_chart",
            "image_width": IMG_WIDTH,
            "image_height": IMG_HEIGHT,
            "ma_periods": list(MA_PERIODS),
            "min_rel_span": min_rel_span_for(TIMEFRAME),
        },
        "buckets": {name: len(selected[name]) for name in buckets},
        "symbols": symbols_present,
        "symbol_count": len(symbols_present),
        "time_coverage": {"first": times[0], "last": times[-1]},
        "total_images": len(samples),
        "samples": samples,
    }

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "review_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    review_ledger.write_review_template(manifest, output / "review_template.jsonl")
    (output / "index.html").write_text(render_index_html(manifest), encoding="utf-8")

    audit = audit_gallery(manifest, duplicate_events=duplicate_events, expected_total=total_images)
    audit["near_duplicate_candidates_rejected"] = len(rejected_duplicates)
    audit["near_duplicate_rejections"] = rejected_duplicates[:50]
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.rmtree(work, ignore_errors=True)
    return {"manifest": manifest, "audit": audit, "output": str(output)}


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def audit_gallery(
    manifest: Mapping[str, Any],
    *,
    duplicate_events: Sequence[Mapping[str, Any]] = (),
    expected_total: int = 240,
    expected_symbols: int = 14,
) -> dict[str, Any]:
    """Check size, bucket balance, coverage, duplicates, and source identity."""
    samples = list(manifest.get("samples", []))
    errors: list[str] = []

    if len(samples) != expected_total:
        errors.append(f"gallery has {len(samples)} images, expected {expected_total}")

    bucket_counts: dict[str, int] = {}
    for sample in samples:
        bucket_counts[sample["bucket"]] = bucket_counts.get(sample["bucket"], 0) + 1
    if sorted(bucket_counts) != sorted(GALLERY_BUCKETS):
        errors.append(f"bucket set mismatch: {sorted(bucket_counts)}")
    per_bucket = expected_total // len(GALLERY_BUCKETS)
    for name, count in sorted(bucket_counts.items()):
        if count != per_bucket:
            errors.append(f"bucket {name} has {count} images, expected {per_bucket}")

    symbols = sorted({sample["symbol"] for sample in samples})
    if len(symbols) < expected_symbols:
        errors.append(f"gallery covers {len(symbols)} symbols, expected {expected_symbols}")
    per_symbol = {symbol: sum(1 for s in samples if s["symbol"] == symbol) for symbol in symbols}
    dominant = max(per_symbol.values()) if per_symbol else 0
    if samples and dominant > max(3 * expected_total // max(len(symbols), 1), 1):
        errors.append("one symbol dominates the gallery")

    review_ids = [sample["review_id"] for sample in samples]
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(set(review_ids)) != len(review_ids):
        errors.append("duplicate review_id")
    if len(set(sample_ids)) != len(sample_ids):
        errors.append("duplicate sample_id")

    endpoints = [(sample["symbol"], sample["window_end_open_time"]) for sample in samples]
    if len(set(endpoints)) != len(endpoints):
        errors.append("duplicate source endpoint")
    groups = [sample["duplicate_group"] for sample in samples]
    duplicate_groups = sorted({group for group in groups if groups.count(group) > 1})
    if duplicate_groups:
        errors.append(f"overlapping window group(s): {', '.join(duplicate_groups)}")

    image_hashes = [sample["image_sha256"] for sample in samples]
    if len(set(image_hashes)) != len(image_hashes):
        errors.append("duplicate image sha256")
    perceptual_pairs = [event for event in duplicate_events if event.get("kind") == "perceptual"]
    if perceptual_pairs:
        errors.append(f"{len(perceptual_pairs)} perceptual near-duplicate pair(s)")

    source_errors = 0
    for sample in samples:
        if sample.get("timeframe") != TIMEFRAME:
            source_errors += 1
        if not sample.get("source_file") or not sample.get("source_sha256"):
            source_errors += 1
        if sample.get("ground_truth") is not None or sample.get("label_status") != "unreviewed":
            source_errors += 1
    leakage_errors = 0
    for sample in samples:
        end_time = pd.Timestamp(sample["window_end_open_time"])
        if end_time >= HOLDOUT_START:
            leakage_errors += 1
    if source_errors:
        errors.append(f"{source_errors} sample(s) failed the source identity check")
    if leakage_errors:
        errors.append(f"{leakage_errors} sample(s) cross the holdout boundary")

    times = sorted(sample["window_end_open_time"] for sample in samples)
    months = sorted({time[:7] for time in times})
    return {
        "schema_version": GALLERY_SCHEMA_VERSION,
        "images": len(samples),
        "buckets": len(bucket_counts),
        "per_bucket": bucket_counts,
        "symbols": len(symbols),
        "symbol_list": symbols,
        "images_per_symbol": per_symbol,
        "time_coverage": {"first": times[0] if times else None, "last": times[-1] if times else None},
        "months_covered": months,
        "duplicates": len(duplicate_events),
        "duplicate_events": list(duplicate_events),
        "source_errors": source_errors,
        "leakage_errors": leakage_errors,
        "unreviewed": sum(1 for sample in samples if sample.get("label_status") == "unreviewed"),
        "valid": not errors,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# blind HTML
# --------------------------------------------------------------------------- #
LEAK_FIELDS = (
    "bucket",
    "symbol",
    "window_end_open_time",
    "legacy_model_conf",
    "legacy_model_key",
    "candidate_mean_full_spread",
)


def render_index_html(manifest: Mapping[str, Any]) -> str:
    """Render the blind review page.

    Only ``review_id``, the image, and the rule-mined candidate box reach the
    page.  Bucket, model, model confidence, symbol, time, and every screening
    statistic stay in the manifest, because a reviewer who can see them is no
    longer blind.

    The page is a one-image-at-a-time keyboard reviewer with a draggable box, not
    a 240-card scroll.  Two reasons: 240 judgements only get made if each costs a
    single keystroke, and a detector needs a *box*, so a positive must ship one.
    The candidate box is a rule proposal to accept, adjust, or ignore — never a
    label, and never a legacy model's prediction.
    """
    samples = list(manifest.get("samples", []))
    payload = [
        {
            "id": str(sample["review_id"]),
            "src": str(sample["image"]),
            "box": sample.get("candidate_box"),
        }
        for sample in samples
    ]
    codes = [
        {"code": code, "key": REASON_CODE_KEYS.get(code)} for code in review_ledger.REASON_CODES
    ]
    return (
        REVIEW_PAGE_TEMPLATE.replace("__SAMPLES__", json.dumps(payload, ensure_ascii=False))
        .replace("__CODES__", json.dumps(codes, ensure_ascii=False))
        .replace("__STATUSES__", json.dumps(list(review_ledger.REVIEW_STATUSES)))
        .replace("__TOTAL__", str(len(samples)))
    )


REASON_CODE_KEYS = {
    "PERFECT_SIX_LINE_DENSE": "p",
    "FAST_ONLY": "f",
    "SLOW_LINES_SEPARATED": "s",
    "SLOPE_TOO_LARGE": "l",
    "DURATION_TOO_SHORT": "d",
    "PRICE_NOT_COMPRESSED": "c",
    "ALREADY_BROKEN_OUT": "b",
    "INCOMPLETE_PATTERN": "i",
    "SCALE_ILLUSION": "z",
    "AMBIGUOUS": "m",
    "BAD_RENDER": "r",
}

REVIEW_PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>盲审 — perfect_ma_dense 5m</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #fbfbfa; --fg: #1b1b19; --dim: #6b6b66; --line: rgba(0,0,0,.12);
  --card: #fff; --pos: #1f9d55; --neg: #d1495b; --unc: #c9922b; --rej: #6b6b66;
  --accent: #2f6fd0; --box: #2f6fd0;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #16171a; --fg: #eceded; --dim: #9a9b9e; --line: rgba(255,255,255,.14);
          --card: #1e2024; --pos: #46c07d; --neg: #ef6b7c; --unc: #e0ab48; --rej: #8b8d92; }
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0; background: var(--bg); color: var(--fg); overflow: hidden;
  font: 14px/1.5 -apple-system, "PingFang SC", "Helvetica Neue", system-ui, sans-serif;
  display: flex; flex-direction: column;
}
header {
  display: flex; align-items: center; gap: 14px; padding: 9px 16px;
  border-bottom: 1px solid var(--line); flex: 0 0 auto;
}
.title { font-weight: 650; white-space: nowrap; }
.title small { font-weight: 400; color: var(--dim); margin-left: 6px; }
.track { flex: 1; height: 6px; border-radius: 99px; background: var(--line); overflow: hidden; min-width: 60px; }
.track > i { display: block; height: 100%; width: 0; background: var(--accent); transition: width .18s; }
.tally { display: flex; gap: 10px; font-variant-numeric: tabular-nums; white-space: nowrap; }
.tally b { font-weight: 600; }
.tally .p { color: var(--pos); } .tally .n { color: var(--neg); }
.tally .u { color: var(--unc); } .tally .r { color: var(--rej); }
.ghost {
  border: 1px solid var(--line); background: transparent; color: inherit;
  border-radius: 8px; padding: 4px 10px; cursor: pointer; font: inherit; white-space: nowrap;
}
.ghost:hover { border-color: var(--accent); }

main { flex: 1 1 auto; min-height: 0; display: flex; align-items: center;
       justify-content: center; padding: 8px 16px; position: relative; }
#frame { position: relative; line-height: 0; box-shadow: 0 1px 3px rgba(0,0,0,.18);
         border-radius: 8px; background: #fff; }
#shot { display: block; max-width: 100%; max-height: 100%; border-radius: 8px; }
main.zoom { overflow: auto; align-items: flex-start; justify-content: flex-start; }
main.zoom #shot { max-width: none; max-height: none; width: 1280px; }
#box {
  position: absolute; border: 2px solid var(--box); border-radius: 2px;
  background: color-mix(in srgb, var(--box) 10%, transparent); cursor: move; display: none;
}
#box.dirty { border-color: var(--unc); background: color-mix(in srgb, var(--unc) 12%, transparent); }
#box .h { position: absolute; width: 12px; height: 12px; background: var(--box);
          border: 1.5px solid #fff; border-radius: 3px; }
#box.dirty .h { background: var(--unc); }
#box .nw { left: -7px; top: -7px; cursor: nwse-resize; }
#box .ne { right: -7px; top: -7px; cursor: nesw-resize; }
#box .sw { left: -7px; bottom: -7px; cursor: nesw-resize; }
#box .se { right: -7px; bottom: -7px; cursor: nwse-resize; }
#rid { position: absolute; top: 12px; left: 22px; font-variant-numeric: tabular-nums;
       color: var(--dim); font-size: 12px; letter-spacing: .5px; }
#flag { position: absolute; top: 10px; right: 22px; font-size: 12px; font-weight: 600; }

footer { flex: 0 0 auto; border-top: 1px solid var(--line); padding: 9px 16px 11px; background: var(--card); }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
#verdicts { display: inline-flex; gap: 8px; flex-wrap: wrap; }
.row + .row { margin-top: 7px; }
.verdict {
  display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; cursor: pointer;
  border: 1.5px solid var(--line); border-radius: 10px; background: transparent;
  color: inherit; font: inherit; font-weight: 550;
}
.verdict[data-v="positive"].on { border-color: var(--pos); background: color-mix(in srgb, var(--pos) 16%, transparent); }
.verdict[data-v="negative"].on { border-color: var(--neg); background: color-mix(in srgb, var(--neg) 16%, transparent); }
.verdict[data-v="uncertain"].on { border-color: var(--unc); background: color-mix(in srgb, var(--unc) 16%, transparent); }
.verdict[data-v="rejected"].on { border-color: var(--rej); background: color-mix(in srgb, var(--rej) 16%, transparent); }
kbd {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px;
  border: 1px solid var(--line); border-bottom-width: 2px; border-radius: 5px;
  padding: 0 5px; color: var(--dim); background: color-mix(in srgb, var(--fg) 5%, transparent);
}
.code {
  display: inline-flex; align-items: center; gap: 6px; font-size: 12px; cursor: pointer;
  border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; user-select: none;
}
.code.on { border-color: var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent); }
#notes { flex: 1 1 240px; min-width: 180px; padding: 6px 10px; border-radius: 8px;
         border: 1px solid var(--line); background: transparent; color: inherit; font: inherit; }
.spacer { flex: 1; }
.hint { color: var(--dim); font-size: 12px; }
#boxstate { font-size: 12px; font-weight: 600; }

dialog { border: 1px solid var(--line); border-radius: 14px; padding: 0; max-width: 580px;
         background: var(--card); color: var(--fg); }
dialog::backdrop { background: rgba(0,0,0,.45); }
.sheet { padding: 18px 22px 20px; }
.sheet h2 { margin: 0 0 4px; font-size: 16px; }
.sheet h3 { margin: 16px 0 6px; font-size: 13px; color: var(--dim); font-weight: 600; }
.keys { display: grid; grid-template-columns: auto 1fr; gap: 6px 14px; align-items: center; }
.sheet ol { margin: 6px 0 0; padding-left: 20px; }
.sheet li { margin: 2px 0; }
#toast { position: fixed; left: 50%; bottom: 96px; transform: translateX(-50%) translateY(8px);
         background: var(--fg); color: var(--bg); padding: 8px 16px; border-radius: 999px;
         opacity: 0; pointer-events: none; transition: opacity .18s, transform .18s; font-size: 13px; z-index: 9; }
#toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }
</style>
</head>
<body>
<header>
  <div class="title">盲审 <small>perfect_ma_dense · 5m · 96 根</small></div>
  <div class="track"><i id="bar"></i></div>
  <div class="tally">
    <span class="p">正 <b id="t-positive">0</b></span>
    <span class="n">负 <b id="t-negative">0</b></span>
    <span class="u">疑 <b id="t-uncertain">0</b></span>
    <span class="r">废 <b id="t-rejected">0</b></span>
    <span class="hint"><b id="t-done">0</b>/__TOTAL__</span>
  </div>
  <label class="hint"><input type="checkbox" id="auto" checked> 自动下一张</label>
  <button class="ghost" id="btn-export">导出 JSONL <kbd>E</kbd></button>
  <button class="ghost" id="btn-help">?</button>
</header>

<main id="stage">
  <span id="rid"></span>
  <span id="flag"></span>
  <div id="frame">
    <img id="shot" alt="">
    <div id="box"><i class="h nw"></i><i class="h ne"></i><i class="h sw"></i><i class="h se"></i></div>
  </div>
</main>

<footer>
  <div class="row">
    <span id="verdicts"></span>
    <span class="spacer"></span>
    <span id="boxstate"></span>
    <button class="ghost" id="btn-resetbox">复原框 <kbd>0</kbd></button>
    <button class="ghost" id="btn-dropbox">删掉框 <kbd>\\</kbd></button>
  </div>
  <div class="row" id="codes"></div>
  <div class="row">
    <input id="notes" placeholder="备注（N 聚焦，Esc 退出）">
    <span class="hint">框：拖动移动 · 拖角缩放 · 在空白处拖出新框 · ← → 翻页 · U 清除 · G 跳到未判 · X 放大</span>
  </div>
</footer>

<dialog id="help"><div class="sheet">
  <h2>这一轮在做什么</h2>
  <p class="hint">图上的蓝框是<strong>规则挖出来的候选</strong>，不是标签，也不是模型预测。
  你的判断就是：这个框圈的东西，是不是非常标准、非常完美的六线均线密集形态。
  框不准就直接拖；框错地方就在空白处拖一个新的；判 positive 必须有框。</p>
  <h3>快捷键</h3>
  <div class="keys">
    <kbd>1</kbd><span>positive — 非常标准（会连框一起记下来）</span>
    <kbd>2</kbd><span>negative — 明确不是</span>
    <kbd>3</kbd><span>uncertain — 说不清（不进训练也不进验证）</span>
    <kbd>4</kbd><span>rejected — 图/数据/渲染有问题</span>
    <kbd>← →</kbd><span>上一张 / 下一张（也可用 K / J，空格 = 下一张）</span>
    <kbd>0</kbd><span>把框复原成候选框</span>
    <kbd>\\</kbd><span>删掉框</span>
    <kbd>U</kbd><span>清除这张的判定</span>
    <kbd>G</kbd><span>跳到第一张未判</span>
    <kbd>X</kbd><span>原尺寸放大 / 还原</span>
    <kbd>N</kbd><span>写备注，<kbd>Esc</kbd> 退出输入</span>
    <kbd>E</kbd><span>导出 JSONL</span>
    <kbd>?</kbd><span>这份说明</span>
  </div>
  <h3>原因代码（可多选，未列出的用鼠标点）</h3>
  <div class="keys" id="codekeys"></div>
  <h3>八条判据 — 全过才是 positive</h3>
  <ol>
    <li>六条线是否全部进入密集，而不是只有快线</li>
    <li>密集是否持续得足够完整，而不是瞬时交叉</li>
    <li>线条是否明显互相靠拢、交织或压缩</li>
    <li>整体斜率是否标准，而不是六线平行单边跑</li>
    <li>价格结构是否同步收缩</li>
    <li>是否尚未发生明显突破</li>
    <li>框的起止是否覆盖真实密集段</li>
    <li>是否足够标准，能无歧义确认</li>
  </ol>
  <p class="hint">犹豫超过几秒就选 uncertain。进度存在浏览器本地，可以分几次做完。</p>
</div></dialog>

<div id="toast"></div>

<script>
const SAMPLES = __SAMPLES__;
const CODES = __CODES__;
const STATUSES = __STATUSES__;
const KEY = "yolo_xx_pr01a_reviews_v3";
const VERDICT_KEYS = { "1": "positive", "2": "negative", "3": "uncertain", "4": "rejected" };
const CODE_BY_KEY = {};
for (const item of CODES) { if (item.key) CODE_BY_KEY[item.key] = item.code; }

let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let cursor = 0;

const stage = document.getElementById("stage");
const frame = document.getElementById("frame");
const shot = document.getElementById("shot");
const boxEl = document.getElementById("box");
const ridEl = document.getElementById("rid");
const flagEl = document.getElementById("flag");
const notesEl = document.getElementById("notes");
const autoEl = document.getElementById("auto");
const helpEl = document.getElementById("help");
const boxState = document.getElementById("boxstate");
const verdictRow = document.getElementById("verdicts");

for (const [key, name] of Object.entries(VERDICT_KEYS)) {
  const button = document.createElement("button");
  button.className = "verdict";
  button.dataset.v = name;
  button.innerHTML = `<kbd>${key}</kbd>${name}`;
  button.addEventListener("click", () => setVerdict(name));
  verdictRow.appendChild(button);
}

const codeRow = document.getElementById("codes");
for (const item of CODES) {
  const label = document.createElement("span");
  label.className = "code";
  label.dataset.code = item.code;
  label.innerHTML = item.key ? `<kbd>${item.key.toUpperCase()}</kbd>${item.code}` : item.code;
  label.addEventListener("click", () => toggleCode(item.code));
  codeRow.appendChild(label);
}
const codeKeys = document.getElementById("codekeys");
for (const item of CODES) {
  if (!item.key) continue;
  const k = document.createElement("kbd");
  k.textContent = item.key.toUpperCase();
  const s = document.createElement("span");
  s.textContent = item.code;
  codeKeys.append(k, s);
}

function current() { return SAMPLES[cursor]; }
function entry(id) {
  if (!state[id]) state[id] = { decision: null, reason_codes: [], box: null, box_touched: false, notes: "" };
  return state[id];
}
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
function boxOf(item, sample) {
  if (item.box === false) return null;          // explicitly removed by the reviewer
  if (Array.isArray(item.box)) return item.box; // moved, resized, or drawn
  return sample.box || null;                    // untouched rule candidate
}

function paintBox(box) {
  if (!box) { boxEl.style.display = "none"; return; }
  const [xc, yc, w, h] = box;
  boxEl.style.display = "block";
  boxEl.style.left = ((xc - w / 2) * 100) + "%";
  boxEl.style.top = ((yc - h / 2) * 100) + "%";
  boxEl.style.width = (w * 100) + "%";
  boxEl.style.height = (h * 100) + "%";
}

function show(index) {
  cursor = Math.max(0, Math.min(SAMPLES.length - 1, index));
  const sample = current();
  const item = entry(sample.id);
  shot.src = sample.src;
  ridEl.textContent = `${sample.id}   ${cursor + 1} / ${SAMPLES.length}`;
  notesEl.value = item.notes || "";
  paintBox(boxOf(item, sample));
  boxEl.classList.toggle("dirty", !!item.box_touched);
  boxState.textContent = boxOf(item, sample)
    ? (item.box_touched ? "框：已调整" : (sample.box ? "框：候选" : "框：新建"))
    : "框：无";
  boxState.style.color = item.box_touched ? "var(--unc)" : "var(--dim)";
  for (const button of verdictRow.querySelectorAll(".verdict")) {
    button.classList.toggle("on", button.dataset.v === item.decision);
  }
  for (const label of codeRow.children) {
    label.classList.toggle("on", (item.reason_codes || []).includes(label.dataset.code));
  }
  flagEl.textContent = item.decision || "";
  flagEl.style.color = item.decision ? `var(--${item.decision.slice(0, 3)})` : "";
  stage.classList.remove("zoom");
  for (let step = 1; step <= 3; step++) {
    const next = SAMPLES[cursor + step];
    if (next) { const pre = new Image(); pre.src = next.src; }
  }
  refresh();
}

function refresh() {
  const counts = { positive: 0, negative: 0, uncertain: 0, rejected: 0 };
  let done = 0;
  for (const sample of SAMPLES) {
    const decision = state[sample.id] && state[sample.id].decision;
    if (decision && counts[decision] !== undefined) { counts[decision]++; done++; }
  }
  for (const status of STATUSES) document.getElementById("t-" + status).textContent = counts[status];
  document.getElementById("t-done").textContent = done;
  document.getElementById("bar").style.width = (done / SAMPLES.length * 100) + "%";
}

function setVerdict(name) {
  const sample = current();
  const item = entry(sample.id);
  item.decision = item.decision === name ? null : name;
  if (item.decision === "positive" && !boxOf(item, sample)) {
    item.box = [0.5, 0.5, 0.25, 0.2];
    item.box_touched = true;
    toast("positive 必须有框——已给一个默认框，拖到位置上");
  }
  save();
  show(cursor);
  if (item.decision && autoEl.checked && cursor < SAMPLES.length - 1) {
    setTimeout(() => show(cursor + 1), 120);
  }
}
function toggleCode(code) {
  const item = entry(current().id);
  const list = new Set(item.reason_codes || []);
  list.has(code) ? list.delete(code) : list.add(code);
  item.reason_codes = [...list];
  save();
  show(cursor);
}
function clearCurrent() { delete state[current().id]; save(); show(cursor); }
function resetBox() {
  const item = entry(current().id);
  item.box = null; item.box_touched = false;
  save(); show(cursor);
}
function dropBox() {
  const item = entry(current().id);
  item.box = false; item.box_touched = true;
  save(); show(cursor);
}
function firstUnreviewed() {
  const index = SAMPLES.findIndex(s => !(state[s.id] && state[s.id].decision));
  show(index === -1 ? cursor : index);
}
function toast(text) {
  const node = document.getElementById("toast");
  node.textContent = text;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 1800);
}

// ---- box editing -------------------------------------------------------- //
let drag = null;
function rect() { return shot.getBoundingClientRect(); }
function norm(event) {
  const r = rect();
  return [
    Math.min(1, Math.max(0, (event.clientX - r.left) / r.width)),
    Math.min(1, Math.max(0, (event.clientY - r.top) / r.height)),
  ];
}
function writeBox(x1, y1, x2, y2) {
  const left = Math.min(x1, x2), right = Math.max(x1, x2);
  const top = Math.min(y1, y2), bottom = Math.max(y1, y2);
  const w = Math.max(right - left, 0.01), h = Math.max(bottom - top, 0.01);
  const item = entry(current().id);
  item.box = [
    +(left + w / 2).toFixed(6), +(top + h / 2).toFixed(6),
    +w.toFixed(6), +h.toFixed(6),
  ];
  item.box_touched = true;
  paintBox(item.box);
  boxEl.classList.add("dirty");
  boxState.textContent = "框：已调整";
  boxState.style.color = "var(--unc)";
}
function corners() {
  const item = entry(current().id);
  const box = boxOf(item, current());
  if (!box) return null;
  const [xc, yc, w, h] = box;
  return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2];
}
frame.addEventListener("mousedown", (event) => {
  const handle = event.target.classList && event.target.classList.contains("h") ? event.target : null;
  const [x, y] = norm(event);
  const box = corners();
  if (handle && box) {
    const anchor = handle.classList.contains("nw") ? [box[2], box[3]]
                 : handle.classList.contains("ne") ? [box[0], box[3]]
                 : handle.classList.contains("sw") ? [box[2], box[1]]
                 : [box[0], box[1]];
    drag = { mode: "resize", anchor };
  } else if (event.target === boxEl && box) {
    drag = { mode: "move", grab: [x, y], start: box.slice() };
  } else {
    drag = { mode: "draw", anchor: [x, y] };
    writeBox(x, y, x, y);
  }
  event.preventDefault();
});
window.addEventListener("mousemove", (event) => {
  if (!drag) return;
  const [x, y] = norm(event);
  if (drag.mode === "draw" || drag.mode === "resize") {
    writeBox(drag.anchor[0], drag.anchor[1], x, y);
  } else if (drag.mode === "move") {
    const dx = x - drag.grab[0], dy = y - drag.grab[1];
    const [x1, y1, x2, y2] = drag.start;
    const w = x2 - x1, h = y2 - y1;
    const nx = Math.min(Math.max(x1 + dx, 0), 1 - w);
    const ny = Math.min(Math.max(y1 + dy, 0), 1 - h);
    writeBox(nx, ny, nx + w, ny + h);
  }
});
window.addEventListener("mouseup", () => { if (drag) { drag = null; save(); } });

function exportJsonl() {
  const stamp = new Date().toISOString().replace(/\\.\\d+Z$/, "Z");
  const lines = [];
  let missingBox = 0;
  for (const sample of SAMPLES) {
    const item = state[sample.id];
    if (!item || !item.decision) continue;
    const box = boxOf(item, sample);
    let action = "none", adjusted = null;
    if (item.box_touched && box) { action = "adjust"; adjusted = box; }
    else if (!item.box_touched && box && item.decision === "positive") { action = "accept"; }
    if (item.decision === "positive" && !box) missingBox++;
    lines.push(JSON.stringify({
      review_id: sample.id,
      decision: item.decision,
      reason_codes: item.reason_codes || [],
      box_action: action,
      adjusted_box: adjusted,
      reviewer: "owner",
      reviewed_at: stamp,
      notes: item.notes || ""
    }));
  }
  if (!lines.length) { toast("还没有任何判定"); return; }
  const url = URL.createObjectURL(new Blob([lines.join("\\n") + "\\n"], { type: "application/x-ndjson" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = "owner_reviews_pr01a.jsonl";
  link.click();
  URL.revokeObjectURL(url);
  toast(missingBox ? `已导出 ${lines.length} 条，其中 ${missingBox} 个 positive 没有框` : `已导出 ${lines.length} 条`);
}

notesEl.addEventListener("input", () => { entry(current().id).notes = notesEl.value; save(); });
document.getElementById("btn-export").addEventListener("click", exportJsonl);
document.getElementById("btn-help").addEventListener("click", () => helpEl.showModal());
document.getElementById("btn-resetbox").addEventListener("click", resetBox);
document.getElementById("btn-dropbox").addEventListener("click", dropBox);

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName);
  if (event.key === "Escape") { document.activeElement.blur(); return; }
  if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (VERDICT_KEYS[event.key]) { event.preventDefault(); setVerdict(VERDICT_KEYS[event.key]); return; }
  if (CODE_BY_KEY[key]) { event.preventDefault(); toggleCode(CODE_BY_KEY[key]); return; }
  if (key === "arrowright" || key === "j" || key === " ") { event.preventDefault(); show(cursor + 1); return; }
  if (key === "arrowleft" || key === "k") { event.preventDefault(); show(cursor - 1); return; }
  if (key === "0") { event.preventDefault(); resetBox(); return; }
  if (key === "\\\\") { event.preventDefault(); dropBox(); return; }
  if (key === "u") { event.preventDefault(); clearCurrent(); return; }
  if (key === "g") { event.preventDefault(); firstUnreviewed(); return; }
  if (key === "x") { event.preventDefault(); stage.classList.toggle("zoom"); return; }
  if (key === "n") { event.preventDefault(); notesEl.focus(); return; }
  if (key === "e") { event.preventDefault(); exportJsonl(); return; }
  if (event.key === "?" || key === "h") { event.preventDefault(); helpEl.showModal(); return; }
});

firstUnreviewed();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yolo-xx-pattern", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-spec", help="validate the pattern spec and print its digest")
    validate.add_argument("--spec", default="configs/PERFECT_PATTERN_SPEC_V1.yaml")
    validate.add_argument("--require-frozen", action="store_true")

    build = sub.add_parser("build-owner-gallery", help="build the blind Owner review gallery")
    build.add_argument("--spec", default="configs/PERFECT_PATTERN_SPEC_V1.yaml")
    build.add_argument("--registry", default=DEFAULT_REGISTRY)
    build.add_argument("--out", default=DEFAULT_OUT)
    build.add_argument("--repo", default=".")
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--scan-pool-per-symbol", type=int, default=120)
    build.add_argument("--device", default="mps")
    build.add_argument("--work-dir", default=None)
    build.add_argument(
        "--scores-cache",
        default=None,
        help="reuse a previous legacy-model scan for the same pool and weights",
    )

    audit = sub.add_parser("audit-reviews", help="audit an Owner review file against a gallery")
    audit.add_argument("--manifest", default=f"{DEFAULT_OUT}/review_manifest.json")
    audit.add_argument("--reviews", default=None)
    audit.add_argument("--out", default=None)

    args = parser.parse_args(argv)

    if args.command == "validate-spec":
        from .pattern_spec import require_owner_frozen_spec

        spec = (
            require_owner_frozen_spec(args.spec)
            if args.require_frozen
            else load_pattern_spec(args.spec)
        )
        print(json.dumps(spec_summary(spec), indent=2, ensure_ascii=False))
        return 0

    if args.command == "build-owner-gallery":
        result = build_owner_gallery(
            spec_path=args.spec,
            registry_path=args.registry,
            out_dir=args.out,
            repo=args.repo,
            seed=args.seed,
            scan_pool_per_symbol=args.scan_pool_per_symbol,
            device=args.device,
            work_dir=args.work_dir,
            scores_cache=args.scores_cache,
        )
        print(json.dumps(result["audit"], indent=2, ensure_ascii=False))
        return 0 if result["audit"]["valid"] else 1

    return review_ledger.main(
        [
            "--manifest",
            args.manifest,
            *(["--reviews", args.reviews] if args.reviews else []),
            *(["--out", args.out] if args.out else []),
        ]
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
