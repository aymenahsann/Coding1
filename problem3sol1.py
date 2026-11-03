import collections
import heapq
import random
import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# ==========================================
# 1. CORE ALGORITHMS
# ==========================================

def get_path(pred, target):
    """Reconstruct path from source to target using predecessor map."""
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        curr = pred.get(curr)
    return path[::-1] if path and pred.get(path[-1], False) is None else []

def bfs(adj, source):
    dist = {source: 0}
    pred = {source: None}
    queue = collections.deque([source])
    while queue:
        u = queue.popleft()
        for edge in adj.get(u, []):
            v = edge[0] if isinstance(edge, tuple) else edge
            if v not in dist:
                dist[v] = dist[u] + 1
                pred[v] = u
                queue.append(v)
    return dist, {k: get_path(pred, k) for k in dist}

def dijkstra(adj, source):
    dist = {source: 0}
    pred = {source: None}
    pq = [(0, source)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist[u]:
            continue
        for v, w in adj.get(u, []):
            if d_u + w < dist.get(v, float('inf')):
                dist[v] = d_u + w
                pred[v] = u
                heapq.heappush(pq, (dist[v], v))
    return dist, {k: get_path(pred, k) for k in dist}

def bellman_ford(edges, nodes, source):
    dist = {v: float('inf') for v in nodes}
    pred = {v: None for v in nodes}
    dist[source] = 0

    for _ in range(len(nodes) - 1):
        updated = False
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                pred[v] = u
                updated = True
        if not updated:
            break

    # Negative cycle check
    cycle = None
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            curr = v
            for _ in range(len(nodes)):
                curr = pred[curr]
            cycle = [curr]
            step = pred[curr]
            while step != curr:
                cycle.append(step)
                step = pred[step]
            cycle.append(curr)
            cycle = cycle[::-1]
            break

    paths = {k: get_path(pred, k) for k in dist if dist[k] != float('inf')}
    return dist, paths, cycle

# ==========================================
# 2. VALIDATION (100 Random Graphs)
# ==========================================

def run_validation(seed=42):
    random.seed(seed)
    bfs_errs, dijk_errs = 0, 0
    for i in range(100):
        n = random.randint(10, 40)
        G = nx.erdos_renyi_graph(n, 0.2, directed=True, seed=seed + i)
        adj = {u: [] for u in range(n)}
        for u, v in G.edges():
            w = random.randint(1, 10)
            G[u][v]['weight'] = w
            adj[u].append((v, w))
        s = random.randint(0, n - 1)

        b_dist, _ = bfs(adj, s)
        nx_b = nx.single_source_shortest_path_length(G, s)
        if b_dist != nx_b:
            bfs_errs += 1

        d_dist, _ = dijkstra(adj, s)
        nx_d = nx.single_source_dijkstra_path_length(G, s, weight='weight')
        if d_dist != nx_d:
            dijk_errs += 1

    print(f"Validation: BFS disagreements = {bfs_errs}, Dijkstra disagreements = {dijk_errs}")

# ==========================================
# 3. BENCHMARKING & LOG-LOG PLOTTING
# ==========================================

def run_benchmarks(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    def measure(n_vals, m_fn):
        t_bfs, t_dijk = [], []
        for n in n_vals:
            m = m_fn(n)
            G = nx.gnm_random_graph(n, m, directed=True, seed=seed)
            adj = {u: [] for u in range(n)}
            for u, v in G.edges():
                adj[u].append((v, 1))

            t0 = time.perf_counter()
            bfs(adj, 0)
            t_bfs.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            dijkstra(adj, 0)
            t_dijk.append(time.perf_counter() - t0)
        return t_bfs, t_dijk

    n_r1 = np.logspace(2, 4, 5, dtype=int)
    bfs_r1, dijk_r1 = measure(n_r1, lambda n: min(4 * n, n * (n - 1)))

    n_r2 = np.logspace(2, 3.2, 5, dtype=int)
    bfs_r2, dijk_r2 = measure(n_r2, lambda n: int(0.01 * n * (n - 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, x, b, d, title in [
        (ax1, n_r1, bfs_r1, dijk_r1, "Regime 1: Mean Degree = 4"),
        (ax2, n_r2, bfs_r2, dijk_r2, "Regime 2: Density = 0.01")
    ]:
        b_slope = np.polyfit(np.log10(x), np.log10(b), 1)[0]
        d_slope = np.polyfit(np.log10(x), np.log10(d), 1)[0]
        ax.loglog(x, b, 'o-', label=f"BFS (slope={b_slope:.2f})")
        ax.loglog(x, d, 's-', label=f"Dijkstra (slope={d_slope:.2f})")
        ax.set_title(title)
        ax.set_xlabel("n")
        ax.set_ylabel("Time (s)")
        ax.grid(True, which="both", ls="--")
        ax.legend()

    plt.tight_layout()
    plt.show()

# ==========================================
# 4. METABOLIC NETWORK ANALYSIS
# ==========================================

def run_metabolic_analysis():
    raw_edges = [
        ("Glucose", "Glucose-6-Phosp", 1),
        ("Glucose-6-Phosp", "Fructose-6-Phos", 0),
        ("Fructose-6-Phos", "Fructose-1,6-Bis", 1),
        ("Fructose-1,6-Bis", "Glyceraldehyde-", 0),
        ("Fructose-1,6-Bis", "Dihydroxyaceton", 0),
        ("Glyceraldehyde-", "1,3-Bisphosphog", 0),
        ("1,3-Bisphosphog", "3-Phosphoglycer", -1),
        ("3-Phosphoglycer", "2-Phosphoglycer", 0),
        ("2-Phosphoglycer", "Phosphoenolpyr", 0),
        ("Phosphoenolpyr", "Pyruvate", -1),
        ("Pyruvate", "Lactate", 0),
        ("Acetyl-CoA", "Pyruvate", 0),
        ("Acetyl-CoA", "Citrate", 0),
        ("Citrate", "Isocitrate", 0),
        ("Isocitrate", "a-Ketoglutarate", 0),
        ("a-Ketoglutarate", "Succinyl-CoA", 0),
        ("Succinyl-CoA", "Succinate", -1),
        ("Fumarate", "Succinate", 0),
        ("Fumarate", "Malate", 0),
        ("Oxaloacetate", "Malate", 0),
        ("Oxaloacetate", "Phosphoenolpyr", 1),
        ("Oxaloacetate", "Aspartate", 0),
        ("a-Ketoglutarate", "Glutamate", 0),
        ("3-Phosphoglycer", "Serine", 0),
        ("Alanine", "Pyruvate", 0)
    ]

    adj = collections.defaultdict(list)
    nodes = set()
    for u, v, w in raw_edges:
        adj[u].append((v, w))
        nodes.update([u, v])

    targets = ["Pyruvate", "Oxaloacetate", "Serine"]

    # Dijkstra
    dijk_dist, _ = dijkstra(adj, "Glucose")
    print("\n--- Dijkstra from Glucose ---")
    for t in targets:
        print(f"{t}: {dijk_dist.get(t, 'Unreachable')}")

    # Bellman-Ford
    bf_dist, _, cycle = bellman_ford(raw_edges, nodes, "Glucose")
    print("\n--- Bellman-Ford from Glucose ---")
    for t in targets:
        val = bf_dist[t]
        print(f"{t}: {val if val != float('inf') else 'Unreachable'}")

    print(f"\nNegative cycle reachable: {cycle is not None}")
    if cycle:
        print("Cycle:", " -> ".join(cycle))

if __name__ == "__main__":
    run_validation()
    run_metabolic_analysis()
    run_benchmarks()