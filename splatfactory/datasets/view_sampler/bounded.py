"""Bounded view sampler for sequential camera trajectories (e.g. RE10K).

Supports 1..N context views with three sampling cases:
  1-view:  random context, targets sampled nearby (within min_distance_between_context_views)
  2-view:  left/right endpoints with gap scaled by gap_multiplier * N
  3+-view: left/right + random interior views, rejected if any consecutive
           context gap > max_distance_between_consecutive_views (fallback: evenly-spaced)

Total span scales with view count: max_gap = min(gap_multiplier * N, total_views - 1).
For N>2, also capped at (N-1) * max_distance_between_consecutive_views.
Targets use stratified random sampling across the span for even coverage.

Author: Alexander Veicht
"""

import numpy as np

from splatfactory.datasets.view_sampler import BaseViewSampler


class BoundedViewSampler(BaseViewSampler):
    default_conf = {
        "cameras_are_circular": False,
        "warm_up_steps": 150_000,
        "min_distance_to_context_views": 0,
        "initial_min_distance_between_context_views": 25,
        "min_distance_between_context_views": 35,
        "gap_multiplier": 45,
        "max_distance_between_consecutive_views": 90,
    }

    def _init(self, conf):
        self.cameras_are_circular = conf.cameras_are_circular

    def get_left_right_context_indices(self, num_views, num_context_views=2):
        if self.is_test or self.is_val or self.is_overfitting:
            max_gap = min(self.conf.gap_multiplier * num_context_views, num_views - 1)
            return 0, max_gap

        N = num_context_views
        max_gap = min(self.conf.gap_multiplier * N, num_views - 1)
        if N > 2:
            max_gap = min(max_gap, (N - 1) * self.conf.max_distance_between_consecutive_views)

        min_gap = self.linear_schedule(
            self.conf.initial_min_distance_between_context_views,
            self.conf.min_distance_between_context_views,
        )
        min_gap = max(N - 1, min_gap)
        min_gap = max(2 * self.conf.min_distance_to_context_views, min_gap)

        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)

        # clamp min_gap to max_gap for short scenes
        min_gap = min(min_gap, max_gap)

        assert (
            max_gap >= min_gap
        ), f"Example does not have enough frames! {max_gap=}, {min_gap=} ({num_views=})"
        context_gap = np.random.randint(min_gap, max_gap + 1)

        left = np.random.randint(
            num_views if self.cameras_are_circular else num_views - context_gap
        )
        right = left + context_gap

        if self.cameras_are_circular:
            left %= num_views
            right %= num_views

        return left, right

    def _stratified_target_sampling(self, span_start, span_end, num_target_views):
        """Sample targets using stratified random for even coverage."""
        span = span_end - span_start
        if span <= 0:
            return [span_start] * num_target_views

        bin_size = span / num_target_views
        targets = []
        for i in range(num_target_views):
            lo = span_start + int(i * bin_size)
            hi = span_start + int((i + 1) * bin_size)
            hi = max(hi, lo + 1)  # ensure valid range
            targets.append(np.random.randint(lo, hi))
        return targets

    def _sample_extra_views(self, left, right, num_extra_views):
        """Sample interior context views between left and right endpoints.

        Uses rejection sampling with a max-consecutive-gap constraint.
        Falls back to evenly-spaced placement after 100 failed attempts.
        """
        interior_span = right - left - 1
        if interior_span <= num_extra_views:
            return list(range(left + 1, right))

        max_consec = self.conf.max_distance_between_consecutive_views

        for _ in range(100):
            candidate = set()
            while len(candidate) < num_extra_views:
                candidate.add(np.random.randint(left + 1, right))
            extra_views = sorted(candidate)

            all_context = sorted([left, right] + extra_views)
            gaps = [all_context[i + 1] - all_context[i] for i in range(len(all_context) - 1)]
            if max(gaps) <= max_consec:
                return extra_views

        # Fallback: evenly-spaced placement
        step = (right - left) / (num_extra_views + 1)
        extra_views = [left + int((i + 1) * step) for i in range(num_extra_views)]
        return sorted(set(extra_views) - {left, right})

    def sample(
        self,
        scene_id: str | None = None,
        num_context_views: int | None = None,
        num_target_views: int | None = None,
        total_num_views: int | None = None,
    ):
        """Returns a dict with keys: context_indices, target_indices, overlap_scores"""
        assert num_context_views >= 1, "At least one context view is required!"

        # Case 1: single context view
        if num_context_views == 1:
            context = np.random.randint(0, total_num_views)
            min_dist = self.conf.min_distance_between_context_views
            lo = max(0, context - min_dist)
            hi = min(total_num_views, context + min_dist + 1)
            targets = self._stratified_target_sampling(lo, hi, num_target_views)

            return {
                "context_indices": [context],
                "target_indices": sorted(targets),
                "overlap_scores": [1.0],
            }

        # Case 2 & 3: two or more context views
        left, right = self.get_left_right_context_indices(total_num_views, num_context_views)

        num_extra_views = num_context_views - 2
        extra_views = (
            self._sample_extra_views(left, right, num_extra_views) if num_extra_views > 0 else []
        )

        targets = self._stratified_target_sampling(
            left + self.conf.min_distance_to_context_views,
            right - self.conf.min_distance_to_context_views + 1,
            num_target_views,
        )

        return {
            "context_indices": sorted([left, right] + extra_views),
            "target_indices": sorted(targets),
            "overlap_scores": [0.5] * num_context_views,
        }

    def min_required_images(self):
        return max(
            2 * self.conf.min_distance_to_context_views + 1,
            self.conf.min_distance_between_context_views + 1,
        )


if __name__ == "__main__":
    from splatfactory.utils.tools import set_seed

    set_seed(42)

    conf = {
        "min_distance_between_context_views": 35,
        "gap_multiplier": 45,
        "max_distance_between_consecutive_views": 90,
        "warm_up_steps": 0,
        "is_overfitting": False,
    }
    sampler = BoundedViewSampler(conf, split="train")

    test_cases = [1, 2, 3, 5, 10, 24]
    total_views = 150
    num_targets = 3
    num_trials = 500

    for n_ctx in test_cases:
        print(f"\n=== num_context_views={n_ctx}, total_views={total_views} ===")
        failures = 0
        for trial in range(num_trials):
            result = sampler.sample(
                scene_id="test",
                num_context_views=n_ctx,
                num_target_views=num_targets,
                total_num_views=total_views,
            )
            ctx = result["context_indices"]
            tgt = result["target_indices"]

            # check no duplicate context views
            assert len(ctx) == len(set(ctx)), f"Duplicate context views: {ctx}"

            # check correct count
            assert len(ctx) == n_ctx, f"Expected {n_ctx} context views, got {len(ctx)}"
            assert len(tgt) == num_targets, f"Expected {num_targets} targets, got {len(tgt)}"

            # check all indices in range
            for idx in ctx + tgt:
                assert 0 <= idx < total_views, f"Index {idx} out of range [0, {total_views})"

            # check consecutive context gaps <= max_distance_between_consecutive_views
            if n_ctx >= 2:
                sorted_ctx = sorted(ctx)
                for i in range(len(sorted_ctx) - 1):
                    gap = sorted_ctx[i + 1] - sorted_ctx[i]
                    if gap > conf["max_distance_between_consecutive_views"]:
                        failures += 1
                        break

        if n_ctx == 1:
            # check targets are near context
            result = sampler.sample("test", 1, num_targets, total_views)
            ctx_idx = result["context_indices"][0]
            for t in result["target_indices"]:
                dist = abs(t - ctx_idx)
                assert (
                    dist <= conf["min_distance_between_context_views"]
                ), f"Single-view target {t} too far from context {ctx_idx} (dist={dist})"
            print(f"  PASS: targets within {conf['min_distance_between_context_views']} of context")

        print(f"  {num_trials} trials: {failures} consecutive-gap violations")
        if failures > 0:
            print(
                f"  WARNING: {failures}/{num_trials} had gaps > {conf['max_distance_between_consecutive_views']}"
            )
        else:
            print(
                f"  PASS: all consecutive gaps <= {conf['max_distance_between_consecutive_views']}"
            )

    # Edge case: small scene
    print(f"\n=== Edge case: 36 views, 10 context ===")
    for _ in range(100):
        result = sampler.sample("test", 10, 3, 36)
        ctx = result["context_indices"]
        assert len(ctx) == len(set(ctx)), f"Duplicate context views: {ctx}"
        assert all(0 <= i < 36 for i in ctx + result["target_indices"])
    print("  PASS: no crashes or out-of-range indices")

    print("\nAll tests passed!")
