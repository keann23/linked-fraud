"""Rebuild every output the story notebook and README use, in order.

    python src/pipeline.py            # skip steps whose outputs already exist
    python src/pipeline.py --force    # rebuild everything (about 30 to 60 minutes on a laptop)

Needs data/raw/train_transaction.csv and data/raw/train_identity.csv.
"""
from __future__ import annotations

import sys
import time

import pandas as pd

import compare
import features
import graph
import identity
import load
import model
import patterns
import viz
import hero

P = model.PROCESSED


def step(name: str, outputs: list, fn, force: bool) -> None:
    if not force and all(o.exists() for o in outputs):
        print(f"[skip] {name}")
        return
    t = time.time()
    print(f"[run ] {name} ...", flush=True)
    fn()
    print(f"[done] {name} in {time.time() - t:.0f}s", flush=True)


def baseline_model() -> None:
    import lightgbm as lgb
    df = model.load_frame()
    train, test = model.split(df)
    _, models = model.run_baselines(df)
    m, cols, scores = models["+ velocity features"]
    model.save_predictions(test, scores, "baseline")
    m.save_model(str(P / "baseline.txt"))
    imp = model.shap_importance(lgb.Booster(model_file=str(P / "baseline.txt")), test[cols])
    imp.to_csv(P / "shap_baseline.csv")


def network_comparison() -> None:
    df = compare.load_frame()
    _, test = model.split(df)
    scores = compare.train_scores(df)
    r, m = compare.compare(test, scores)
    r.to_csv(P / "phase5_results.csv")
    m.to_csv(P / "phase5_missed.csv")
    compare.coverage(test, scores).to_csv(P / "phase5_coverage.csv")


def figures() -> None:
    con = identity.connect()
    viz.save(identity.plot_purity(con, identity.compare_groupings(con)), "02_identity_purity")
    imp = pd.read_csv(P / "shap_baseline.csv", index_col=0).iloc[:, 0]
    viz.save(model.plot_shap(imp, *model.shap_title(imp)), "03_baseline_shap")
    c = pd.read_parquet(P / "cluster_table.parquet")
    viz.save(graph.plot_size_vs_fraud(c), "04_cluster_size_vs_fraud")
    df = compare.load_frame()
    _, test = model.split(df)
    scores = pd.read_parquet(compare.RESULTS_PARQUET)
    r, _ = compare.compare(test, scores)
    viz.save(compare.plot_headline(r, compare.coverage(test, scores)), "05_headline_comparison")
    gcon = graph.connect()
    viz.save(patterns.plot_coordination(patterns.coordination(patterns.ring_frame(gcon))), "06_ring_coordination")
    viz.save(hero.plot_hero(gcon), "01_hero_ring")


def main(force: bool = False) -> None:
    step("1 load raw CSVs to parquet", [load.BASE_PARQUET], lambda: load.build_base(force=True), force)
    step("2 rebuild identities", [identity.IDENTITY_PARQUET], lambda: identity.build_identity(force=True), force)
    step("3a velocity features (SQL)", [features.VELOCITY_PARQUET], lambda: features.build_velocity(force=True), force)
    step("3b baseline model + SHAP", [P / "predictions.parquet", P / "baseline.txt", P / "shap_baseline.csv"],
         baseline_model, force)
    step("4 network and clusters", [graph.CLUSTERS_PARQUET, graph.EDGES_PARQUET, P / "cluster_table.parquet"],
         graph.run, force)
    step("5a as-of network features (SQL)", [features.NETWORK_PARQUET], lambda: features.build_network(force=True), force)
    step("5b model vs rings, 3 seeds", [compare.RESULTS_PARQUET, P / "phase5_results.csv"], network_comparison, force)
    step("7 figures", [], figures, True)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
