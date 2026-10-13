from __future__ import annotations

import fairvision_staged_fairness_pipeline as sfp


def _patch_race_subgroup_names() -> None:
    original_apply = sfp.apply_demographic_mode_defaults

    def patched_apply(cfg: sfp.Config) -> sfp.Config:
        cfg = original_apply(cfg)
        if str(cfg.demographic_mode).strip().lower() == "race":
            cfg.subgroup_names = ("asian", "black", "white")
        return cfg

    sfp.apply_demographic_mode_defaults = patched_apply


if __name__ == "__main__":
    _patch_race_subgroup_names()
    sfp.run_entry(sfp.parse_args())
