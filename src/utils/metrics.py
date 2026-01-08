"""Ball tracking metrics computation and display."""

from typing import List

import numpy as np


def compute_ball_metrics(
    ball_tracks: List[dict],
    ball_debug: List[dict] | None = None,
    conf_threshold: float | None = None,
) -> dict:
    """Compute comprehensive ball tracking metrics.

    Args:
        ball_tracks: List of frame dicts with ball detections
        ball_debug: Optional debug info from ball tracker
        conf_threshold: Optional confidence threshold for metrics

    Returns:
        Dictionary of computed metrics
    """
    total_frames = len(ball_tracks)
    present_flags = []
    observed_flags = []
    predicted_flags = []
    confs = []
    sizes = []
    centers = []

    for frame_idx, frame_dict in enumerate(ball_tracks):
        if 1 not in frame_dict:
            present_flags.append(0)
            observed_flags.append(0)
            predicted_flags.append(0)
            continue

        present_flags.append(1)
        info = frame_dict[1]
        interpolated = bool(info.get("interpolated", False))
        predicted = bool(info.get("predicted", False))
        observed_flags.append(0 if interpolated else 1)
        predicted_flags.append(1 if predicted else 0)

        bbox = info.get("bbox")
        if bbox is not None:
            width = max(0.0, bbox[2] - bbox[0])
            height = max(0.0, bbox[3] - bbox[1])
            sizes.append(float(np.sqrt(width * height)))
            centers.append((frame_idx, ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)))

        conf = info.get("confidence")
        if conf is not None and not interpolated:
            confs.append(float(conf))

    present_count = int(sum(present_flags))
    observed_count = int(sum(observed_flags))
    predicted_count = int(sum(predicted_flags))
    interp_count = max(0, present_count - observed_count)

    miss_streaks = []
    current = 0
    for flag in present_flags:
        if flag == 0:
            current += 1
        elif current:
            miss_streaks.append(current)
            current = 0
    if current:
        miss_streaks.append(current)

    observed_segments = []
    current = 0
    for flag in observed_flags:
        if flag == 1:
            current += 1
        elif current:
            observed_segments.append(current)
            current = 0
    if current:
        observed_segments.append(current)

    speeds = []
    velocities = []
    for (idx_a, center_a), (idx_b, center_b) in zip(centers[:-1], centers[1:]):
        dt = max(1, idx_b - idx_a)
        delta = np.array(center_b) - np.array(center_a)
        vel = delta / float(dt)
        speeds.append(float(np.linalg.norm(vel)))
        velocities.append(vel)

    accelerations = []
    for prev, curr in zip(speeds[:-1], speeds[1:]):
        accelerations.append(float(abs(curr - prev)))

    jerks = []
    for prev, curr in zip(accelerations[:-1], accelerations[1:]):
        jerks.append(float(abs(curr - prev)))

    turn_angles = []
    for prev, curr in zip(velocities[:-1], velocities[1:]):
        prev_norm = float(np.linalg.norm(prev))
        curr_norm = float(np.linalg.norm(curr))
        if prev_norm < 1e-6 or curr_norm < 1e-6:
            continue
        cos_theta = float(np.clip(np.dot(prev, curr) / (prev_norm * curr_norm), -1.0, 1.0))
        turn_angles.append(float(np.degrees(np.arccos(cos_theta))))

    size_cv = None
    if sizes:
        mean_size = float(np.mean(sizes))
        if mean_size > 0:
            size_cv = float(np.std(sizes) / mean_size)

    conf_below_rate = None
    if conf_threshold is not None and confs:
        below = sum(1 for c in confs if c < conf_threshold)
        conf_below_rate = below / len(confs) * 100.0

    candidate_mean = None
    candidate_p95 = None
    ambiguity_rate = None
    candidate_post_gate_mean = None
    candidate_post_gate_p95 = None
    ambiguity_post_gate_rate = None
    reject_rate = None
    reject_conf_rate = None
    reject_aspect_rate = None
    reject_gate_rate = None
    select_rate = None
    reject_aspect_total = None
    reject_area_total = None
    reject_jump_total = None
    reject_acquire_total = None

    if ball_debug:
        raw_counts = [d.get("raw_count", 0) for d in ball_debug]
        post_conf_counts = [d.get("post_conf", 0) for d in ball_debug]
        post_aspect_counts = [d.get("post_aspect", 0) for d in ball_debug]
        post_gate_counts = [d.get("post_gate", 0) for d in ball_debug]
        selected_counts = [d.get("selected", 0) for d in ball_debug]
        candidates = [c for c in raw_counts if c > 0]
        if candidates:
            candidate_mean = float(np.mean(candidates))
            candidate_p95 = float(np.percentile(candidates, 95))
            ambiguity_rate = sum(1 for c in candidates if c > 1) / len(candidates) * 100.0
            reject = sum(
                1 for raw, post in zip(raw_counts, post_gate_counts) if raw > 0 and post == 0
            )
            reject_rate = reject / len(candidates) * 100.0
        post_gate_candidates = [c for c in post_gate_counts if c > 0]
        if post_gate_candidates:
            candidate_post_gate_mean = float(np.mean(post_gate_candidates))
            candidate_post_gate_p95 = float(np.percentile(post_gate_candidates, 95))
            ambiguity_post_gate_rate = (
                sum(1 for c in post_gate_candidates if c > 1) / len(post_gate_candidates) * 100.0
            )

        raw_total = int(np.sum(raw_counts))
        post_conf_total = int(np.sum(post_conf_counts))
        post_aspect_total = int(np.sum(post_aspect_counts))
        post_gate_total = int(np.sum(post_gate_counts))
        selected_total = int(np.sum(selected_counts))

        if raw_total:
            reject_conf_rate = (raw_total - post_conf_total) / raw_total * 100.0
        if post_conf_total:
            reject_aspect_rate = (post_conf_total - post_aspect_total) / post_conf_total * 100.0
        if post_aspect_total:
            reject_gate_rate = (post_aspect_total - post_gate_total) / post_aspect_total * 100.0
        if post_gate_total:
            select_rate = selected_total / post_gate_total * 100.0

        reject_aspect_total = int(sum(d.get("reject_aspect", 0) for d in ball_debug))
        reject_area_total = int(sum(d.get("reject_area", 0) for d in ball_debug))
        reject_jump_total = int(sum(d.get("reject_jump", 0) for d in ball_debug))
        reject_acquire_total = int(sum(d.get("reject_acquire", 0) for d in ball_debug))

    metrics = {
        "total_frames": total_frames,
        "present_frames": present_count,
        "present_pct": (present_count / total_frames * 100.0) if total_frames else 0.0,
        "observed_frames": observed_count,
        "observed_pct": (observed_count / total_frames * 100.0) if total_frames else 0.0,
        "interpolated_frames": interp_count,
        "interpolated_pct_of_present": (interp_count / present_count * 100.0) if present_count else 0.0,
        "predicted_frames": predicted_count,
        "predicted_pct_of_present": (predicted_count / present_count * 100.0) if present_count else 0.0,
        "observed_segments": len(observed_segments),
        "observed_segment_mean": float(np.mean(observed_segments)) if observed_segments else 0.0,
        "conf_mean": float(np.mean(confs)) if confs else None,
        "conf_p50": float(np.median(confs)) if confs else None,
        "conf_below_rate": conf_below_rate,
        "bbox_size_mean": float(np.mean(sizes)) if sizes else None,
        "bbox_size_p50": float(np.median(sizes)) if sizes else None,
        "bbox_size_cv": size_cv,
        "miss_streaks": miss_streaks,
        "miss_streak_max": int(max(miss_streaks)) if miss_streaks else 0,
        "miss_streak_mean": float(np.mean(miss_streaks)) if miss_streaks else 0.0,
        "speed_mean": float(np.mean(speeds)) if speeds else None,
        "speed_p95": float(np.percentile(speeds, 95)) if speeds else None,
        "accel_p95": float(np.percentile(accelerations, 95)) if accelerations else None,
        "jerk_p95": float(np.percentile(jerks, 95)) if jerks else None,
        "turn_p50": float(np.median(turn_angles)) if turn_angles else None,
        "turn_p95": float(np.percentile(turn_angles, 95)) if turn_angles else None,
        "candidate_mean": candidate_mean,
        "candidate_p95": candidate_p95,
        "ambiguity_rate": ambiguity_rate,
        "candidate_post_gate_mean": candidate_post_gate_mean,
        "candidate_post_gate_p95": candidate_post_gate_p95,
        "ambiguity_post_gate_rate": ambiguity_post_gate_rate,
        "reject_rate": reject_rate,
        "reject_conf_rate": reject_conf_rate,
        "reject_aspect_rate": reject_aspect_rate,
        "reject_gate_rate": reject_gate_rate,
        "select_rate": select_rate,
        "reject_aspect_total": reject_aspect_total,
        "reject_area_total": reject_area_total,
        "reject_jump_total": reject_jump_total,
        "reject_acquire_total": reject_acquire_total,
    }
    return metrics


def print_ball_metrics(metrics: dict, label: str = "Ball") -> None:
    """Display ball metrics in formatted output.

    Args:
        metrics: Dictionary of computed metrics
        label: Label prefix for output
    """
    print("\nBall metrics:")
    print(
        f"{label}: present={metrics['present_frames']}/{metrics['total_frames']}"
        f" ({metrics['present_pct']:.1f}%), observed={metrics['observed_frames']}"
        f" ({metrics['observed_pct']:.1f}%), interpolated={metrics['interpolated_frames']}"
        f" ({metrics['interpolated_pct_of_present']:.1f}% of present)"
    )
    print(
        f"  predicted frames: {metrics['predicted_frames']}"
        f" ({metrics['predicted_pct_of_present']:.1f}% of present)"
    )
    if metrics["observed_segments"]:
        print(
            f"  observed segments: count={metrics['observed_segments']},"
            f" mean_len={metrics['observed_segment_mean']:.1f}"
        )
    if metrics["conf_mean"] is not None:
        print(f"  conf: mean={metrics['conf_mean']:.3f}, p50={metrics['conf_p50']:.3f}")
        if metrics["conf_below_rate"] is not None:
            print(f"  conf below threshold: {metrics['conf_below_rate']:.1f}%")
    if metrics["bbox_size_mean"] is not None:
        print(
            f"  bbox size: mean={metrics['bbox_size_mean']:.2f},"
            f" p50={metrics['bbox_size_p50']:.2f}"
        )
        if metrics["bbox_size_cv"] is not None:
            print(f"  bbox size cv: {metrics['bbox_size_cv']:.3f}")
    if metrics["miss_streak_max"]:
        print(
            f"  miss streaks: count={len(metrics['miss_streaks'])},"
            f" max={metrics['miss_streak_max']},"
            f" mean={metrics['miss_streak_mean']:.1f}"
        )
    if metrics["speed_mean"] is not None:
        print(
            f"  speed(px): mean={metrics['speed_mean']:.2f},"
            f" p95={metrics['speed_p95']:.2f}"
        )
        if metrics["accel_p95"] is not None:
            print(f"  accel p95: {metrics['accel_p95']:.2f}, jerk p95: {metrics['jerk_p95']:.2f}")
        if metrics["turn_p50"] is not None:
            print(f"  turn angle: p50={metrics['turn_p50']:.1f}, p95={metrics['turn_p95']:.1f}")
    if metrics["candidate_mean"] is not None:
        print(
            f"  candidates: mean={metrics['candidate_mean']:.2f},"
            f" p95={metrics['candidate_p95']:.2f},"
            f" ambiguity={metrics['ambiguity_rate']:.1f}%,"
            f" reject={metrics['reject_rate']:.1f}%"
        )
        if metrics["candidate_post_gate_mean"] is not None:
            print(
                f"  post-gate candidates: mean={metrics['candidate_post_gate_mean']:.2f},"
                f" p95={metrics['candidate_post_gate_p95']:.2f},"
                f" ambiguity={metrics['ambiguity_post_gate_rate']:.1f}%"
            )
        if metrics["reject_conf_rate"] is not None:
            print(
                f"  gate drop: conf={metrics['reject_conf_rate']:.1f}%,"
                f" aspect={metrics['reject_aspect_rate']:.1f}%,"
                f" gate={metrics['reject_gate_rate']:.1f}%,"
                f" select={metrics['select_rate']:.1f}%"
            )
        if metrics["reject_aspect_total"] is not None:
            print(
                f"  gate fails (non-exclusive): aspect={metrics['reject_aspect_total']},"
                f" area={metrics['reject_area_total']},"
                f" jump={metrics['reject_jump_total']},"
                f" acquire={metrics['reject_acquire_total']}"
            )
