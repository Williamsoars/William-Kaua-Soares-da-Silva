import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from collections import Counter

# ------------------------------
#  Utility Functions
# ------------------------------

def gcd(a, b):
    return math.gcd(a, b)

def lcm(a, b):
    return a // gcd(a, b) * b

def lcm_list(values):
    from functools import reduce
    return reduce(lcm, values, 1)

def prime_factors(n):
    """Return prime factorization as dict {prime: exponent}."""
    i = 2
    factors = {}
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 1 if i == 2 else 2  # skip evens after 2
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def divisors_from_factors(factors):
    primes = list(factors.items())
    if not primes:
        return [1]

    def gen(idx):
        if idx == len(primes):
            yield 1
        else:
            p, e = primes[idx]
            for d in gen(idx+1):
                mul = 1
                for k in range(e+1):
                    yield d * mul
                    mul *= p
    return sorted(gen(0))

def divisors(n):
    return divisors_from_factors(prime_factors(n))

# ------------------------------
#  Modular Graph Construction
# ------------------------------

def build_modular_graph(primes, A):
    """Adjacency matrix for exponent A."""
    n = len(primes)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        p = primes[i]
        for j in range(n):
            if i == j:
                continue
            q = primes[j]
            if pow(p, A, q) == 1:
                adj[i, j] = 1
            if pow(q, A, p) == 1:
                adj[j, i] = 1
    return adj

def compute_density(adj):
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    edges = int(adj.sum())
    total = n * (n - 1)
    return edges / total

def compute_vertex_zeta(primes, A):
    return [math.gcd(p - 1, A) / (p - 1) for p in primes]

# ------------------------------
#  Complete Statistical Report
# ------------------------------

def full_statistics(df, outdir):
    from scipy.stats import pearsonr, spearmanr, kendalltau, skew, kurtosis
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.linear_model import LinearRegression

    density = df["density"].values
    avg_z = df["avg_vertex_zeta"].values
    graph_z = df["graph_zeta"].values

    # Residuals
    residuals = density - avg_z

    # Linear regression (density ~ avg_z)
    X = avg_z.reshape(-1, 1)
    reg = LinearRegression().fit(X, density)
    R2 = reg.score(X, density)
    RMSE = math.sqrt(mean_squared_error(density, reg.predict(X)))
    MAE = mean_absolute_error(density, reg.predict(X))

    report_path = os.path.join(outdir, "statistics_report.txt")

    with open(report_path, "w") as f:

        f.write("=== FULL STATISTICAL REPORT ===\n\n")

        f.write("Correlation statistics:\n")
        f.write(f"  Pearson(density, avg_zeta)   = {pearsonr(density, avg_z)[0]:.6f}\n")
        f.write(f"  Spearman(density, avg_zeta)  = {spearmanr(density, avg_z)[0]:.6f}\n")
        f.write(f"  Kendall(density, avg_zeta)   = {kendalltau(density, avg_z)[0]:.6f}\n\n")

        f.write("Regression statistics (density ~ avg_zeta):\n")
        f.write(f"  Intercept = {reg.intercept_:.6f}\n")
        f.write(f"  Slope     = {reg.coef_[0]:.6f}\n")
        f.write(f"  R^2       = {R2:.6f}\n")
        f.write(f"  RMSE      = {RMSE:.6f}\n")
        f.write(f"  MAE       = {MAE:.6f}\n\n")

        f.write("Distribution statistics:\n")
        f.write("  Density:\n")
        f.write(f"    mean   = {np.mean(density):.6f}\n")
        f.write(f"    std    = {np.std(density):.6f}\n")
        f.write(f"    skew   = {skew(density):.6f}\n")
        f.write(f"    kurt   = {kurtosis(density):.6f}\n\n")

        f.write("  Average Zeta:\n")
        f.write(f"    mean   = {np.mean(avg_z):.6f}\n")
        f.write(f"    std    = {np.std(avg_z):.6f}\n")
        f.write(f"    skew   = {skew(avg_z):.6f}\n")
        f.write(f"    kurt   = {kurtosis(avg_z):.6f}\n\n")

        f.write("  Graph Zeta:\n")
        f.write(f"    mean   = {np.mean(graph_z):.6f}\n")
        f.write(f"    std    = {np.std(graph_z):.6f}\n")
        f.write(f"    skew   = {skew(graph_z):.6f}\n")
        f.write(f"    kurt   = {kurtosis(graph_z):.6f}\n\n")

        f.write("Residual statistics (density - avg_zeta):\n")
        f.write(f"    mean   = {np.mean(residuals):.6f}\n")
        f.write(f"    std    = {np.std(residuals):.6f}\n")
        f.write(f"    skew   = {skew(residuals):.6f}\n")
        f.write(f"    kurt   = {kurtosis(residuals):.6f}\n\n")

    return report_path

# ------------------------------
#  Main Analysis Function
# ------------------------------

def analyze_primes(primes, max_divisors=500, demo=False):

    primes = sorted(int(p) for p in primes)
    group_sizes = [p - 1 for p in primes]
    L = lcm_list(group_sizes)

    if demo:
        print("Primes:", primes)
        print("LCM =", L)

    divs = divisors(L)

    if len(divs) > max_divisors:
        idx = np.linspace(0, len(divs)-1, max_divisors, dtype=int)
        divs = [divs[i] for i in idx]

    records = []

    for A in divs:
        adj = build_modular_graph(primes, A)
        density = compute_density(adj)
        vzetas = compute_vertex_zeta(primes, A)
        avg = np.mean(vzetas)
        prod = np.prod(vzetas)

        records.append({
            "A": A,
            "density": density,
            "avg_vertex_zeta": avg,
            "graph_zeta": prod
        })

    df = pd.DataFrame(records).sort_values("A")

    outdir = "modular_zeta_analysis_output"
    os.makedirs(outdir, exist_ok=True)

    # ------------------------------
    # Plots
    # ------------------------------

    # 1. Density vs Avg Zeta
    plt.figure(figsize=(6,4))
    plt.scatter(df["avg_vertex_zeta"], df["density"])
    plt.xlabel("Average Vertex Zeta")
    plt.ylabel("Density")
    plt.title("Density vs Avg Vertex Zeta")
    fig1_path = os.path.join(outdir, "density_vs_avg_vertex_zeta.png")
    plt.savefig(fig1_path)
    plt.close()

    # 2. Density vs Graph Zeta
    plt.figure(figsize=(6,4))
    plt.scatter(df["graph_zeta"], df["density"])
    plt.xlabel("Graph Zeta (product)")
    plt.ylabel("Density")
    plt.title("Density vs Graph Zeta")
    fig2_path = os.path.join(outdir, "density_vs_graph_zeta.png")
    plt.savefig(fig2_path)
    plt.close()

    # 3. Residual plot
    res = df["density"] - df["avg_vertex_zeta"]
    plt.figure(figsize=(6,4))
    plt.hist(res, bins=30)
    plt.title("Residuals: density - avg_zeta")
    fig3_path = os.path.join(outdir, "residuals_histogram.png")
    plt.savefig(fig3_path)
    plt.close()

    # ------------------------------
    # Captions file
    # ------------------------------

    cap_path = os.path.join(outdir, "figure_captions.txt")
    with open(cap_path, "w") as f:
        f.write(
            "Figure 1 — Density vs Average Vertex Zeta:\n"
            "Scatter plot comparing graph density with the mean vertex zeta value.\n\n"
        )
        f.write(
            "Figure 2 — Density vs Graph Zeta (product):\n"
            "Scatter plot comparing graph density with the product of vertex zeta values.\n\n"
        )
        f.write(
            "Figure 3 — Residual Histogram:\n"
            "Distribution of residuals (density − avg_vertex_zeta).\n\n"
        )

    # ------------------------------
    # CSV + Full Statistical Report
    # ------------------------------

    csv_path = os.path.join(outdir, "modular_zeta_summary.csv")
    df.to_csv(csv_path, index=False)

    stats_path = full_statistics(df, outdir)

    return {
        "df": df,
        "plots": {
            "density_vs_avg_vertex_zeta": fig1_path,
            "density_vs_graph_zeta": fig2_path,
            "residuals_histogram": fig3_path
        },
        "captions": cap_path,
        "stats_report": stats_path,
        "csv": csv_path,
        "LCM": L
    }

# ------------------------------
# Demo
# ------------------------------

if __name__ == "__main__":
    primes = [19, 71, 101, 163, 503, 577, 967, 991]
    result = analyze_primes(primes, max_divisors=300, demo=True)
    print("Output directory:", os.path.abspath("modular_zeta_analysis_output"))
