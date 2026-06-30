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
import pyRAPL
import pynvml
import threading

import math

import torch

try:
    from torch_cluster import radius_graph
except Exception as e:
    radius_graph = None


class PowerMeter:
    def __init__(self, device: torch.device, sample_period_s: float = 0.01):
        self.device = device
        self.sample_period_s = sample_period_s

        self.working_cpu_setup = False
        self.working_gpu_setup = False
        self.gpu_handle = None

        if device.type == "cpu" and pyRAPL is not None:
            try:
                pyRAPL.setup()
                self.working_cpu_setup = True
            except Exception as e:
                print(f"pyRAPL unavailable: {e}")

        if device.type == "cuda" and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.working_gpu_setup = True
            except Exception as e:
                print(f"NVML unavailable: {e}")

    def measure(self, fn):
        """
        Returns:
            output, energy_j, avg_power_w, elapsed_ms
        """

        # CPU: use pyRAPL
        if self.device.type == "cpu" and self.working_cpu_setup:
            meter = pyRAPL.Measurement("batch")

            t0 = time.perf_counter()
            meter.begin()
            fn()
            meter.end()
            t1 = time.perf_counter()

            result = meter.result

            pkg_energy = sum(result.pkg) if result.pkg else 0.0
            dram_energy = sum(result.dram) if result.dram else 0.0
            energy = (pkg_energy + dram_energy) / 1e6   # pyRAPL energy is in microjoules -> convert to joules
            elapsed_s = t1 - t0
            avg_power = energy / elapsed_s if elapsed_s > 0 else 0.0
            return avg_power

        # GPU: sample NVML power
        if self.device.type == "cuda" and self.working_gpu_setup:
            samples = []
            stop = threading.Event()

            def sampler():
                while not stop.is_set():
                    power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0
                    samples.append(power)
                    time.sleep(self.sample_period_s)

            torch.cuda.synchronize()
            th = threading.Thread(target=sampler)
            th.start()
            fn()
            torch.cuda.synchronize()
            stop.set()
            th.join()

            avg_power = sum(samples) / len(samples)
            return avg_power

        # Fallback: no energy measurement
        out = fn()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return 0.0


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
):
    if radius_graph is None:
        raise RuntimeError("torch_cluster.radius_graph could not be imported. Install torch-cluster.")
    
    power_meter = PowerMeter(device)
    power_list = []

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

            if device.type == "cuda":
                x = None

                def _h2d():
                    nonlocal x
                    x = x_cpu_b.to(device, non_blocking=False)

                _h2d()
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

            _prep()
            assert x_cont is not None and x_cat is not None and etaphi is not None and batch is not None

            edge_index = None

            def _graph():
                nonlocal edge_index
                edge_index = radius_graph(etaphi, r=deltaR, batch=batch, loop=False, max_num_neighbors=max_neighbors)

            _graph()
            assert edge_index is not None
            if device.type == "cuda":
                assert edge_index.is_cuda, "radius_graph returned CPU edge_index on CUDA backend"

            E_total = int(edge_index.size(1))

            # Forward
            out = None

            def _fwd():
                nonlocal out
                out = model(x_cont, x_cat, edge_index, batch)

            power = power_meter.measure(_fwd)
            power_list.append(power)

            assert out is not None

            if include_d2h and device.type == "cuda":
                def _d2h():
                    _ = out.detach().cpu()

                _d2h()

            sample_id += 1
    return sum(power_list) / len(power_list)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=str, default=".", help="Path to repo root containing net.py, data_loader.py, etc.")
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset directory used by data_loader.fetch_dataloader")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g., .../last.pth.tar)")
    ap.add_argument("--is_delphes", type=int, default=1, help="1 for Delphes PDG mapping, 0 otherwise")
    ap.add_argument("--deltaR", type=float, default=0.4, help="radius_graph cutoff")
    ap.add_argument("--num_graphs", type=int, default=16384, help="Number of measured graphs (excluding warmup)")
    ap.add_argument("--warmup", type=int, default=128, help="Warmup graphs per backend (excluded)")
    ap.add_argument("--max_neighbors", type=int, default=255, help="max_num_neighbors for radius_graph")
    ap.add_argument("--include_d2h", type=int, default=1, help="Include D2H(output) in transfer on GPU")
    ap.add_argument("--run_cpu", type=int, default=1, help="Run CPU backends")
    ap.add_argument("--run_gpu", type=int, default=1, help="Run GPU backends (requires CUDA)")
    ap.add_argument("--run_compile", type=int, default=1, help="Also run torch.compile variants")

    args = ap.parse_args()

    add_repo_to_path(args.repo_root)

    try:
        import model.data_loader as data_loader # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Could not import project data_loader. "
            "Run from your repo root or pass --repo_root pointing to it."
        ) from e

    dataloaders = data_loader.fetch_dataloader(
        data_dir=args.data_dir,
        batch_size=1,
        validation_split=0.2,
        is_delphes=bool(args.is_delphes),
    )
    test_dl = dataloaders["test"]

    num_needed = int(args.warmup) + int(args.num_graphs)
    graphs = prefetch_graphs(test_dl, num_needed)

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
        model = build_net(device=device, is_delphes=bool(args.is_delphes))
        load_checkpoint_into_model(args.ckpt, model)

        if do_compile:
            try:
                model = torch.compile(model, mode="default")
            except Exception as e:
                raise RuntimeError(f"torch.compile failed for backend {name} (batch=1): {e}")

        curr_avg_power = run_backend(
            name=name,
            device=device,
            model=model,
            graphs=graphs,
            deltaR=float(args.deltaR),
            batch_size=1,
            warmup=int(args.warmup),
            num_graphs=int(args.num_graphs),
            include_d2h=bool(args.include_d2h),
            max_neighbors=int(args.max_neighbors),
        )

        print(f"Backend: {name}, Device: {device}, Batch size: 1, Average Power: {curr_avg_power}")

if __name__ == "__main__":
    main()