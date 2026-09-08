from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import traceback

import math

import torch

try:
    from torch_cluster import radius_graph
except Exception as e:
    radius_graph = None


# -----------------------------
# Timing helpers
# -----------------------------

@dataclass
class Timing:
    h2d_ms: float = 0.0
    prep_ms: float = 0.0
    graph_ms: float = 0.0
    forward_ms: float = 0.0
    d2h_ms: float = 0.0

    @property
    def compute_ms(self) -> float:
        return self.prep_ms + self.forward_ms

    @property
    def transfer_ms(self) -> float:
        return self.h2d_ms + self.d2h_ms

    @property
    def e2e_ms(self) -> float:
        return self.compute_ms + self.transfer_ms


class DeviceTimer:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = (device.type == "cuda")

    def timeit(self, fn):
        """Run fn() and return (result, elapsed_ms)."""
        if not self.is_cuda:
            t0 = time.perf_counter_ns()
            out = fn()
            t1 = time.perf_counter_ns()
            return out, (t1 - t0) / 1e6

        # CUDA
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize()
        return out, float(start.elapsed_time(end))

# -----------------------------
# Stats helpers
# -----------------------------

def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if c == f:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p50": float(percentile(values, 50)),
        "p90": float(percentile(values, 90)),
        "p99": float(percentile(values, 99)),
        "max": float(max(values)),
    }


# -----------------------------
# Model / checkpoint loading
# -----------------------------

def add_repo_to_path(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def load_checkpoint_into_model(ckpt_path: str, model: torch.nn.Module) -> None:
    ckpt_path = os.path.expanduser(ckpt_path)

    try:
        import utils
        if hasattr(utils, "load_checkpoint"):
            utils.load_checkpoint(ckpt_path, model)
            return
    except Exception:
        pass

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt['state_dict'], strict=True)


def build_net(device: torch.device, is_delphes: bool) -> torch.nn.Module:
    scale_momentum = 128
    scaling = [1.0 / scale_momentum, 1.0 / scale_momentum, 1.0 / scale_momentum, 1.0, 1.0, 1.0]
    norm = torch.tensor(scaling, dtype=torch.float32, device=device)

    import model.net as net

    model = net.Net(continuous_dim=6, categorical_dim=2, norm=norm, is_delphes=is_delphes)
    model.to(device)
    model.eval()
    return model


# -----------------------------
# Data prep
# -----------------------------

def prefetch_graphs(test_dl, num_needed: int) -> List[Dict[str, torch.Tensor]]:

    graphs: List[Dict[str, torch.Tensor]] = []
    for data in test_dl:
        x = data.x.detach().clone().cpu()
        graphs.append({"x": x})
        if len(graphs) >= num_needed:
            break
    if len(graphs) < num_needed:
        print(f"[WARN] Only prefetched {len(graphs)} graphs (requested {num_needed}).")
    return graphs


def parse_batch_sweep(spec: str) -> List[int]:
    spec = (spec or "").strip()
    if not spec:
        return [1]
    out: List[int] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            b = int(tok)
        except ValueError as e:
            raise ValueError(f"Invalid --batch_sweep token '{tok}'. Use comma-separated ints.") from e
        if b <= 0:
            raise ValueError(f"Invalid batch size {b}. Must be > 0.")
        out.append(b)
    seen = set()
    uniq: List[int] = []
    for b in out:
        if b not in seen:
            uniq.append(b)
            seen.add(b)
    return uniq if uniq else [1]


# -----------------------------
# Benchmark core
# -----------------------------

def run_backend(
    *,
    name: str,
    device: torch.device,
    model: torch.nn.Module,
    graphs: List[Dict[str, torch.Tensor]],
    deltaR: float,
    batch_size: int,
    warmup: int,
    num_graphs: int,
    include_d2h: bool,
    max_neighbors: int = 255,
) -> List[Dict[str, Any]]:
    if radius_graph is None:
        raise RuntimeError("torch_cluster.radius_graph could not be imported. Install torch-cluster.")

    timer = DeviceTimer(device)

    results: List[Dict[str, Any]] = []

    def build_batch(start: int, count: int) -> Tuple[torch.Tensor, List[int]]:
        xs = []
        sizes: List[int] = []
        for j in range(count):
            xj = graphs[start + j]["x"]
            xs.append(xj)
            sizes.append(int(xj.size(0)))
        x_cpu_b = torch.cat(xs, dim=0).contiguous()
        return x_cpu_b, sizes

    idx = 0
    warm_left = min(int(warmup), len(graphs))
    with torch.inference_mode():
        while warm_left > 0:
            take = min(int(batch_size), warm_left)
            x_cpu_b, sizes = build_batch(idx, take)
            idx += take
            warm_left -= take

            if device.type == "cuda":
                x = x_cpu_b.to(device, non_blocking=False)
            else:
                x = x_cpu_b

            # Prep
            x_cont = x[:, :6].clone()
            x_cat = x[:, 6:8].to(dtype=torch.long)
            etaphi = torch.stack([x[:, 3], x[:, 4]], dim=1)
            # Build batch vector on-device from sizes
            sizes_t = torch.tensor(sizes, dtype=torch.long, device=x.device)
            batch = torch.repeat_interleave(
                torch.arange(take, dtype=torch.long, device=x.device),
                sizes_t,
            )

            edge_index = radius_graph(etaphi, r=deltaR, batch=batch, loop=False, max_num_neighbors=max_neighbors)
            if device.type == "cuda":
                assert edge_index.is_cuda, "radius_graph returned CPU edge_index on CUDA backend"

            out = model(x_cont, x_cat, edge_index, batch)
            if include_d2h and device.type == "cuda":
                _ = out.detach().cpu()

    meas_left = min(int(num_graphs), max(0, len(graphs) - idx))
    sample_id = 0
    with torch.inference_mode():
        while meas_left > 0:
            take = min(int(batch_size), meas_left)
            x_cpu_b, sizes = build_batch(idx, take)
            idx += take
            meas_left -= take

            t = Timing()

            if device.type == "cuda":
                x = None

                def _h2d():
                    nonlocal x
                    x = x_cpu_b.to(device, non_blocking=False)

                _, t.h2d_ms = timer.timeit(_h2d)
                assert x is not None
            else:
                x = x_cpu_b

            N_total = int(x_cpu_b.size(0))

            x_cont = None
            x_cat = None
            etaphi = None
            batch = None

            def _prep():
                nonlocal x_cont, x_cat, etaphi, batch
                x_cont = x[:, :6].clone()  # avoid in-place mutation issues
                x_cat = x[:, 6:8].to(dtype=torch.long)
                etaphi = torch.stack([x[:, 3], x[:, 4]], dim=1)
                sizes_t = torch.tensor(sizes, dtype=torch.long, device=x.device)
                batch = torch.repeat_interleave(
                    torch.arange(take, dtype=torch.long, device=x.device),
                    sizes_t,
                )

            _, t.prep_ms = timer.timeit(_prep)
            assert x_cont is not None and x_cat is not None and etaphi is not None and batch is not None

            edge_index = None

            def _graph():
                nonlocal edge_index
                edge_index = radius_graph(etaphi, r=deltaR, batch=batch, loop=False, max_num_neighbors=max_neighbors)

            _, t.graph_ms = timer.timeit(_graph)
            assert edge_index is not None
            if device.type == "cuda":
                assert edge_index.is_cuda, "radius_graph returned CPU edge_index on CUDA backend"

            E_total = int(edge_index.size(1))

            # Forward
            out = None

            def _fwd():
                nonlocal out
                out = model(x_cont, x_cat, edge_index, batch)

            _, t.forward_ms = timer.timeit(_fwd)
            assert out is not None

            if include_d2h and device.type == "cuda":
                def _d2h():
                    _ = out.detach().cpu()

                _, t.d2h_ms = timer.timeit(_d2h)

            graphs_in_batch = int(take)
            e2e_ms_per_graph = t.e2e_ms / graphs_in_batch
            compute_ms_per_graph = t.compute_ms / graphs_in_batch
            transfer_ms_per_graph = t.transfer_ms / graphs_in_batch
            graphs_per_s = graphs_in_batch / (t.e2e_ms / 1e3) if t.e2e_ms > 0 else float("inf")
            nodes_per_s = N_total / (t.e2e_ms / 1e3) if t.e2e_ms > 0 else float("inf")

            results.append(
                {
                    "backend": name,
                    "batch_size": int(batch_size),
                    "graphs_in_batch": graphs_in_batch,
                    "sample": sample_id,
                    "N_total": N_total,
                    "E_total": E_total,
                    "h2d_ms": t.h2d_ms,
                    "prep_ms": t.prep_ms,
                    "graph_ms": t.graph_ms,
                    "forward_ms": t.forward_ms,
                    "d2h_ms": t.d2h_ms,
                    "compute_ms": t.compute_ms,
                    "transfer_ms": t.transfer_ms,
                    "e2e_ms": t.e2e_ms,
                    "compute_ms_per_graph": compute_ms_per_graph,
                    "transfer_ms_per_graph": transfer_ms_per_graph,
                    "e2e_ms_per_graph": e2e_ms_per_graph,
                    "graphs_per_s": graphs_per_s,
                    "nodes_per_s": nodes_per_s,
                }
            )
            sample_id += 1

    return results


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Path to repo root containing net.py, data_loader.py, etc.")
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset directory used by data_loader.fetch_dataloader")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g., last.pth.tar)")
    ap.add_argument("--is_delphes", type=int, default=1, help="1 for Delphes PDG mapping, 0 otherwise")
    ap.add_argument("--deltaR", type=float, default=0.4, help="radius_graph cutoff")
    ap.add_argument("--num_graphs", type=int, default=16384, help="Number of measured graphs (excluding warmup)")
    ap.add_argument("--warmup", type=int, default=128, help="Warmup graphs per backend (excluded)")
    ap.add_argument(
        "--batch_sweep",
        type=str,
        default='1,2,4,8,16',
        help="Optional comma-separated micro-batch sizes to sweep (e.g., '1,2,4,8,16'). Empty => batch=1 only.",
    )
    ap.add_argument("--max_neighbors", type=int, default=255, help="max_num_neighbors for radius_graph")
    ap.add_argument("--include_d2h", type=int, default=1, help="Include D2H(output) time in transfer on GPU")
    ap.add_argument("--run_cpu", type=int, default=1, help="Run CPU backends")
    ap.add_argument("--run_gpu", type=int, default=1, help="Run GPU backends (requires CUDA)")
    ap.add_argument("--run_compile", type=int, default=1, help="Also run torch.compile variants")
    ap.add_argument("--outdir", type=str, default="bench_out", help="Output directory")

    args = ap.parse_args()

    add_repo_to_path(args.repo_root)

    try:
        import model.data_loader as data_loader # type: ignore
    except Exception as e:
        traceback.print_exc()
        raise

    dataloaders = data_loader.fetch_dataloader(
        data_dir=args.data_dir,
        batch_size=1,
        validation_split=0.2,
        is_delphes=bool(args.is_delphes),
    )
    test_dl = dataloaders["test"]

    batch_sizes = parse_batch_sweep(args.batch_sweep)
    num_needed = int(args.warmup) + int(args.num_graphs)
    graphs = prefetch_graphs(test_dl, num_needed)

    os.makedirs(args.outdir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    backends_to_run: List[Tuple[str, torch.device, bool]] = []  # (name, device, compile)

    if args.run_cpu:
        backends_to_run.append(("cpu_eager", torch.device("cpu"), False))
        if args.run_compile:
            backends_to_run.append(("cpu_compile", torch.device("cpu"), True))

    if args.run_gpu:
        if not torch.cuda.is_available():
            print("[WARN] --run_gpu=1 but CUDA not available; skipping GPU backends.")
        else:
            backends_to_run.append(("gpu_eager", torch.device("cuda"), False))
            if args.run_compile:
                backends_to_run.append(("gpu_compile", torch.device("cuda"), True))

    for name, device, do_compile in backends_to_run:
        for bsz in batch_sizes:
            model = build_net(device=device, is_delphes=bool(args.is_delphes))
            load_checkpoint_into_model(args.ckpt, model)

            if do_compile:
                try:
                    model = torch.compile(model, mode="default")
                except Exception as e:
                    raise RuntimeError(f"torch.compile failed for backend {name} (batch={bsz}): {e}")

            rows = run_backend(
                name=name,
                device=device,
                model=model,
                graphs=graphs,
                deltaR=float(args.deltaR),
                batch_size=int(bsz),
                warmup=int(args.warmup),
                num_graphs=int(args.num_graphs),
                include_d2h=bool(args.include_d2h),
                max_neighbors=int(args.max_neighbors),
            )
            all_rows.extend(rows)
            gcount = sum(int(r.get("graphs_in_batch", 1)) for r in rows)
            print(f"[OK] {name} (batch={bsz}): collected {len(rows)} batches / {gcount} graphs")

    per_sample_path = os.path.join(args.outdir, "per_sample.csv")
    fieldnames = [
        "backend",
        "batch_size",
        "graphs_in_batch",
        "sample",
        "N_total",
        "E_total",
        "h2d_ms",
        "prep_ms",
        "graph_ms",
        "forward_ms",
        "d2h_ms",
        "compute_ms",
        "transfer_ms",
        "e2e_ms",
        "compute_ms_per_graph",
        "transfer_ms_per_graph",
        "e2e_ms_per_graph",
        "graphs_per_s",
        "nodes_per_s",
    ]
    with open(per_sample_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    by_group: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (str(r.get("backend")), int(r.get("batch_size", 1)))
        by_group.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {}

    for (backend, bsz), rows in sorted(by_group.items(), key=lambda x: (x[0][0], x[0][1])):
        graphs_total = sum(int(x.get("graphs_in_batch", 1)) for x in rows)
        batches_total = len(rows)

        comp_b = [float(x["compute_ms"]) for x in rows]
        tr_b = [float(x["transfer_ms"]) for x in rows]
        e2e_b = [float(x["e2e_ms"]) for x in rows]

        comp_g = [float(x["compute_ms_per_graph"]) for x in rows]
        tr_g = [float(x["transfer_ms_per_graph"]) for x in rows]
        e2e_g = [float(x["e2e_ms_per_graph"]) for x in rows]
        graphs_s = [float(x["graphs_per_s"]) for x in rows]
        nodes_s = [float(x["nodes_per_s"]) for x in rows]
        graph_g = [float(x["graph_ms"]) / float(x.get("graphs_in_batch", 1)) for x in rows]
        fwd_g = [float(x["forward_ms"]) / float(x.get("graphs_in_batch", 1)) for x in rows]

        s = {
            "compute_ms_per_batch": summarize(comp_b),
            "transfer_ms_per_batch": summarize(tr_b),
            "e2e_ms_per_batch": summarize(e2e_b),
            "compute_ms_per_graph": summarize(comp_g),
            "transfer_ms_per_graph": summarize(tr_g),
            "e2e_ms_per_graph": summarize(e2e_g),
            "graph_ms_per_graph": summarize(graph_g),
            "forward_ms_per_graph": summarize(fwd_g),
            "graphs_per_s": summarize(graphs_s),
            "nodes_per_s": summarize(nodes_s),
            "counts": {"batches": batches_total, "graphs": graphs_total},
        }
        summary_json.setdefault(backend, {})[str(bsz)] = s

        summary_rows.append(
            {
                "backend": backend,
                "batch_size": bsz,
                "batches": batches_total,
                "graphs": graphs_total,
                "compute_mean_ms_per_graph": s["compute_ms_per_graph"].get("mean", float("nan")),
                "compute_p50_ms_per_graph": s["compute_ms_per_graph"].get("p50", float("nan")),
                "compute_p99_ms_per_graph": s["compute_ms_per_graph"].get("p99", float("nan")),
                "transfer_mean_ms_per_graph": s["transfer_ms_per_graph"].get("mean", float("nan")),
                "e2e_mean_ms_per_graph": s["e2e_ms_per_graph"].get("mean", float("nan")),
                "e2e_p50_ms_per_graph": s["e2e_ms_per_graph"].get("p50", float("nan")),
                "e2e_p99_ms_per_graph": s["e2e_ms_per_graph"].get("p99", float("nan")),
                "graph_mean_ms_per_graph": s["graph_ms_per_graph"].get("mean", float("nan")),
                "forward_mean_ms_per_graph": s["forward_ms_per_graph"].get("mean", float("nan")),
                "throughput_mean_graphs_per_s": s["graphs_per_s"].get("mean", float("nan")),
                "throughput_p50_graphs_per_s": s["graphs_per_s"].get("p50", float("nan")),
                "throughput_p99_graphs_per_s": s["graphs_per_s"].get("p99", float("nan")),
                "nodes_per_s_mean": s["nodes_per_s"].get("mean", float("nan")),
            }
        )

    summary_csv_path = os.path.join(args.outdir, "summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["backend"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    summary_json_path = os.path.join(args.outdir, "summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_json, f, indent=2)

    plot_bars_path = os.path.join(args.outdir, "plot_bars.csv")
    with open(plot_bars_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "backend",
                "batch_size",
                "compute_mean_ms_per_graph",
                "transfer_mean_ms_per_graph",
                "e2e_mean_ms_per_graph",
                "throughput_mean_graphs_per_s",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(
                {
                    "backend": r["backend"],
                    "batch_size": r["batch_size"],
                    "compute_mean_ms_per_graph": r["compute_mean_ms_per_graph"],
                    "transfer_mean_ms_per_graph": r["transfer_mean_ms_per_graph"],
                    "e2e_mean_ms_per_graph": r["e2e_mean_ms_per_graph"],
                    "throughput_mean_graphs_per_s": r["throughput_mean_graphs_per_s"],
                }
            )

    meta = {
        "args": vars(args),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    meta_path = os.path.join(args.outdir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote:\n  {per_sample_path}\n  {summary_csv_path}\n  {summary_json_path}\n  {plot_bars_path}\n  {meta_path}\n")


if __name__ == "__main__":
    main()