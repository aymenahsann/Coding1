import random
import time

# Standard 20-letter amino-acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Naive search
def find_common_substring_naive(x: str, y: str, k: int):
    n = len(x)
    m = len(y)

    # Check if k is invalid or valid
    if k <= 0 or k > n or k > m:
        return None

    # Valid indices range from 0 up to n - k inclusive (total: n - k + 1 positions)
    for i in range(n - k + 1):
        # Extract the window of length k from sequence x starting at index i
        window_x = x[i : i + k]

        # Compare window_x against all valid starting positions in sequence y
        for j in range(m - k + 1):
            window_y = y[j : j + k]

            # In Python, string equality checks up to k characters until a mismatch or match
            if window_x == window_y:
                return window_x  # Return the matching substring immediately

    # If all pairs of windows have been checked without a match, return None
    return None


# Hash set search
def find_common_substring_hash(x: str, y: str, k: int):
    n = len(x)
    m = len(y)
    if k <= 0 or k > n or k > m:
        return None

    # Phase 1: Store all k-mers of x in a hash set
    seen = set()
    for i in range(n - k + 1):
        seen.add(x[i : i + k])

    # Phase 2: Scan y and query membership
    for j in range(m - k + 1):
        window_y = y[j : j + k]
        if window_y in seen:
            return window_y

    return None


def generate_random_protein(length: int) -> str:
    """Generates a random protein sequence of given length using Python's random module."""
    return "".join(random.choices(AMINO_ACIDS, k=length))


def run_benchmarks():
    """Runs the required benchmark on random sequences with fixed seed 42."""
    random.seed(42)

    n_values = [10**3, 10**4, 10**5, 10**6]
    k_values = [3, 10, 50]

    # Upper bound for naive search to prevent indefinite hanging (10^5 * 10^5 = 10^10 pairs)
    NAIVE_MAX_N = 10**4

    benchmark_rows = []

    print("=" * 72)
    print(
        f"{'k':<4} | {'n = m':<8} | {'Hash Time (s)':<15} | {'Naive Time (s)':<15} | {'Status'}"
    )
    print("=" * 72)

    for k in k_values:
        for n in n_values:
            seq_x = generate_random_protein(n)
            seq_y = generate_random_protein(n)

            # Benchmark Hash approach
            t0 = time.perf_counter()
            match_hash = find_common_substring_hash(seq_x, seq_y, k)
            t_hash = time.perf_counter() - t0

            # Benchmark Naive approach
            if n <= NAIVE_MAX_N:
                t0 = time.perf_counter()
                match_naive = find_common_substring_naive(seq_x, seq_y, k)
                t_naive = time.perf_counter() - t0
                status = "Match Found" if match_hash else "No Match"
                naive_str = f"{t_naive:.6f}"
            else:
                t_naive = None
                status = "Omitted (O(n^2) scaling)"
                naive_str = "N/A"

            benchmark_rows.append((k, n, t_hash, t_naive))
            print(
                f"{k:<4} | {n:<8} | {t_hash:<15.6f} | {naive_str:<15} | {status}"
            )

    print("=" * 72)
    return benchmark_rows


def find_fine_grained_crossover(k=10):
    """Scans small values of n to locate the empirical crossover point."""
    random.seed(42)
    print(f"\nSearching for empirical crossover point n* (k = {k})...")

    for n in range(10, 150, 5):
        times_naive = []
        times_hash = []
        for _ in range(5):
            x = generate_random_protein(n)
            y = generate_random_protein(n)

            t0 = time.perf_counter()
            find_common_substring_naive(x, y, k)
            times_naive.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            find_common_substring_hash(x, y, k)
            times_hash.append(time.perf_counter() - t0)

        avg_naive = sum(times_naive) / len(times_naive)
        avg_hash = sum(times_hash) / len(times_hash)

        if avg_hash < avg_naive:
            print(
                f"--> Crossover reached at n ≈ {n}: Naive = {avg_naive:.6f}s, Hash = {avg_hash:.6f}s"
            )
            return n

    return None


if __name__ == "__main__":
    print("--- Mode Selection ---")
    print("1. Interactive input (test custom sequences)")
    print("2. Run full benchmark suite & crossover analysis")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        seq_x = input("Enter first sequence (x): ").strip().upper()
        seq_y = input("Enter second sequence (y): ").strip().upper()

        try:
            k_val = int(input("Enter window length k: ").strip())
        except ValueError:
            print("Error: k must be a valid integer.")
            exit(1)

        result = find_common_substring_naive(seq_x, seq_y, k_val)

        if result is not None:
            print(f"Common substring of length {k_val} found: '{result}'")
        else:
            print(f"No common substring of length {k_val} found.")

    elif choice == "2":
        records = run_benchmarks()
        export_csv_data(records)
        find_fine_grained_crossover(k=10)
    else:
        print("Invalid choice selected.")