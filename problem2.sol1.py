COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def load_fasta_sequence(filepath: str) -> str:
    """Reads a FASTA file and returns the concatenated nucleotide sequence."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or FASTA comment/header lines
            if not line or line.startswith(">"):
                continue
            chunks.append(line.upper())

    return "".join(chunks)


# Pass your local file path here:
filepath = "ecoli_U00096.fasta"  # or full path, e.g., "/path/to/ecoli_U00096.fasta"
ecoli_sequence = load_fasta_sequence(filepath)

print(f"Loaded {len(ecoli_sequence):,} base pairs.")

def is_rev_comp_palindrome(s: str) -> bool:
    """Check if s is equal to its reverse complement."""
    m = len(s)
    if m % 2 != 0:
        return False
    # Two-pointer test from outside in
    left = 0
    right = m - 1
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 'T', 't': 'A', 'c': 'G', 'g': 'C'}
    while left < right:
        if s[left] not in comp or comp[s[left]] != s[right]:
            return False
        left += 1
        right -= 1
    return True


def naive_rc_palindromes(t: str, k: int):
    """
    Find every position at which a reverse-complement palindrome of length >= k begins.
    Yields (start_position, length) tuples.
    """
    n = len(t)
    # Palindromes must have even length; start at smallest even l >= k
    start_l = k if k % 2 == 0 else k + 1

    hits = []
    for i in range(n):
        for l in range(start_l, n - i + 1, 2):
            sub = t[i:i + l]
            if is_rev_comp_palindrome(sub):
                hits.append((i, l))
    return hits


def expand_center_rc_palindromes(t: str, k: int):
    """
    Finds all reverse-complement palindromes of length >= k via center expansion.
    Yields (start_position, length) tuples.
    """
    n = len(t)
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
            'a': 'T', 't': 'A', 'c': 'G', 'g': 'C'}
    hits = []

    # Centers are located between index c-1 and c, for c in 1 .. n-1
    for c in range(1, n):
        left = c - 1
        right = c
        radius = 0

        while left >= 0 and right < n:
            b_left = t[left]
            b_right = t[right]
            if b_left in comp and comp[b_left] == b_right:
                radius += 1
                if 2 * radius >= k:
                    hits.append((left, 2 * radius))
                left -= 1
                right += 1
            else:
                break  # Mismatch encountered; stop expanding this center

    return hits