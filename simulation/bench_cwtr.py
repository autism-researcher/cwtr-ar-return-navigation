"""
CWTR computational-cost benchmark.
Measures real per-fix latency, throughput, peak memory and empirical O(T)
scaling of the CWTR pipeline (confidence scoring + KF/RTS smoother), using the
SAME functions as the paper's reference implementation (cwtr_simulation.py).
All numbers printed here are measured on the host machine; no hand editing.

Reproduces the table "Measured CWTR Runtime and Memory vs. Route Length".
Run from this directory:  python3 bench_cwtr.py
"""
import sys, os, time, tracemalloc, platform, subprocess
import numpy as np

# import the exact reference functions used in the paper (same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cwtr_simulation as C


def make_track(L, rng):
    """A length-L random-walk track + reported accuracy, like gen_obs output."""
    h = rng.uniform(0, 2 * np.pi); p = np.zeros(2); pos = np.zeros((L, 2))
    for t in range(L):
        h += rng.normal(0, 0.15)
        p = p + C.SPEED * C.DT * np.array([np.cos(h), np.sin(h)])
        pos[t] = p
    sigma = 4.0 * np.ones(L)
    z = pos + rng.normal(0, 1, (L, 2)) * sigma[:, None]
    a = np.clip(np.abs(sigma * (1 + rng.normal(0, 0.2, L))), 1.0, None)
    return z, a


def cwtr_once(z, a, L):
    c = C.conf(z, a)
    Rc = [np.eye(2) * ((C.SIGMA0 ** 2 / c[t]) * (C.GATE if c[t] < C.TAU else 1.0))
          for t in range(L)]
    return C.kalman_rts(z, Rc)


def time_pipeline(L, reps, rng):
    z, a = make_track(L, rng)
    cwtr_once(z, a, L)  # warm-up
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cwtr_once(z, a, L)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return ts.mean(), ts.std(), np.median(ts)


if __name__ == "__main__":
    print("Platform:", platform.platform())
    print("Python:", platform.python_version(), "| NumPy:", np.__version__)
    try:
        cpu = subprocess.check_output(
            "grep -m1 'model name' /proc/cpuinfo", shell=True).decode().split(':')[1].strip()
        nproc = subprocess.check_output("nproc", shell=True).decode().strip()
        print("CPU:", cpu, "| cores:", nproc)
    except Exception:
        pass
    print()

    rng = np.random.default_rng(7)
    Ls = [100, 200, 500, 1000, 2000, 5000, 10000]
    reps = {100: 300, 200: 300, 500: 200, 1000: 100, 2000: 60, 5000: 30, 10000: 20}

    print(f"{'T':<8}{'mean_ms':<11}{'median_ms':<12}{'sd_ms':<9}{'us/fix':<9}{'fixes/s':<10}")
    rows = []
    for L in Ls:
        mean, sd, med = time_pipeline(L, reps[L], rng)
        rows.append((L, mean, med, sd))
        print(f"{L:<8}{mean*1e3:<11.3f}{med*1e3:<12.3f}{sd*1e3:<9.3f}"
              f"{mean/L*1e6:<9.2f}{L/mean:<10.0f}")

    Larr = np.array([r[0] for r in rows], float)
    Tarr = np.array([r[1] for r in rows], float)
    A = np.vstack([Larr, np.ones_like(Larr)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, Tarr, rcond=None)
    pred = A @ np.array([slope, intercept])
    r2 = 1 - np.sum((Tarr - pred) ** 2) / np.sum((Tarr - Tarr.mean()) ** 2)
    print(f"\nLinear fit: time(s) = {slope:.3e}*T + {intercept:.3e}   R^2={r2:.5f}")
    print(f"Marginal per-fix slope: {slope*1e6:.3f} us/fix")

    L = 5000
    z, a = make_track(L, rng)
    tracemalloc.start()
    cwtr_once(z, a, L)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\nPeak Python heap T={L}: {peak/1024:.1f} KiB ({peak/L:.1f} bytes/fix)")
    print(f"Algorithmic per-step state (fp64): {(4+16)*8*2} bytes "
          f"(two 4-vec + two 4x4 cov)")
    print(f"Real-time headroom @1Hz: {slope*1e6:.2f} us/fix "
          f"=> {1e6/(slope*1e6):,.0f}x the 1 s/fix budget "
          f"(CPU duty cycle {slope:.2e})")
