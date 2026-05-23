from __future__ import annotations

import pandas as pd


def segment_queue_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vind segment-intervallen waar >=2 treinen tegelijk aanwezig waren.

    Verwacht kolommen: train_id, segment_id, actual_entry, actual_exit.
    Retourneert 1 rij per overlap-paar.
    """
    required = {"train_id", "segment_id", "actual_entry", "actual_exit"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {sorted(missing)}")

    d = df[list(required)].dropna().copy()
    d["actual_entry"] = d["actual_entry"].astype(float)
    d["actual_exit"] = d["actual_exit"].astype(float)

    rows: list[dict] = []
    for seg, part in d.groupby("segment_id"):
        recs = part[["train_id", "actual_entry", "actual_exit"]].to_dict("records")
        recs.sort(key=lambda r: (r["actual_entry"], r["actual_exit"], r["train_id"]))

        for i in range(len(recs)):
            a = recs[i]
            for j in range(i + 1, len(recs)):
                b = recs[j]
                start = max(a["actual_entry"], b["actual_entry"])
                end = min(a["actual_exit"], b["actual_exit"])
                if end > start:
                    rows.append(
                        {
                            "segment_id": seg,
                            "train_a": int(a["train_id"]),
                            "train_b": int(b["train_id"]),
                            "overlap_start": start,
                            "overlap_end": end,
                            "overlap_seconds": end - start,
                            "a_entry": a["actual_entry"],
                            "a_exit": a["actual_exit"],
                            "b_entry": b["actual_entry"],
                            "b_exit": b["actual_exit"],
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "train_a",
                "train_b",
                "overlap_start",
                "overlap_end",
                "overlap_seconds",
                "a_entry",
                "a_exit",
                "b_entry",
                "b_exit",
            ]
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["overlap_seconds", "segment_id"], ascending=[False, True]).reset_index(drop=True)


def solver_change_log(simulator, include_initial: bool = False) -> pd.DataFrame:
    """
    Toon wat de solver per reschedule-iteratie wijzigde in MIP entry/exit.

    Verwacht simulator._solutions: list[Solution].

    Parameters
    ----------
    include_initial : bool, default False
        False: toon enkel wijzigingen t.o.v. de vorige solve (geen eerste
        baseline-rijen met NaN in *_prev / *_delta).
        True: neem ook de eerste solve op als baseline (met NaN in prev/delta).
    """
    solutions = getattr(simulator, "_solutions", None)
    if not solutions:
        return pd.DataFrame(
            columns=[
                "solve_idx",
                "train_id",
                "segment_id",
                "entry_prev",
                "entry_new",
                "entry_delta",
                "exit_prev",
                "exit_new",
                "exit_delta",
                "objective",
                "runtime",
                "status",
            ]
        )

    prev_entry: dict[tuple[int, str], float] = {}
    prev_exit: dict[tuple[int, str], float] = {}
    rows: list[dict] = []

    for idx, sol in enumerate(solutions, start=1):
        entry = sol.entry or {}
        ex = sol.exit or {}

        keys = sorted(set(entry.keys()).union(ex.keys()))
        for key in keys:
            train_id, segment_id = key
            e_new = entry.get(key)
            x_new = ex.get(key)
            e_prev = prev_entry.get(key)
            x_prev = prev_exit.get(key)

            e_delta = None if e_prev is None or e_new is None else e_new - e_prev
            x_delta = None if x_prev is None or x_new is None else x_new - x_prev

            is_initial = e_prev is None and x_prev is None
            changed = (
                is_initial
                or (e_delta is not None and abs(e_delta) > 1e-9)
                or (x_delta is not None and abs(x_delta) > 1e-9)
            )
            if changed:
                if is_initial and not include_initial:
                    continue
                rows.append(
                    {
                        "solve_idx": idx,
                        "train_id": int(train_id),
                        "segment_id": segment_id,
                        "entry_prev": e_prev,
                        "entry_new": e_new,
                        "entry_delta": e_delta,
                        "exit_prev": x_prev,
                        "exit_new": x_new,
                        "exit_delta": x_delta,
                        "objective": getattr(sol, "objective", None),
                        "runtime": getattr(sol, "runtime", None),
                        "status": getattr(sol, "status", None),
                    }
                )

        prev_entry.update(entry)
        prev_exit.update(ex)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    return out.sort_values(["solve_idx", "train_id", "segment_id"]).reset_index(drop=True)