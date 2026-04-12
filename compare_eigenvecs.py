# ==============================================================================
# 0. USER INPUTS
# ==============================================================================
TEST_REAL_FILE = 'found_eigenvectors_real.txt'
TEST_IMAG_FILE = 'found_eigenvectors_imag.txt'

COMPARE_REAL_FILE = 'asym_eigenvectors_real.txt'
COMPARE_IMAG_FILE = 'asym_eigenvectors_imag.txt'
COMPARE_EIGENVALS_FILE = 'asym_eigenvalues.txt'

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import time

import numpy as np

try:
    from multicore_solver import (
        SEARCH_REAL_MIN, SEARCH_REAL_MAX,
        SEARCH_IMAG_MIN, SEARCH_IMAG_MAX,
    )
except ImportError:
    print("WARNING: Could not import multicore_solver. Using infinite bounds for testing.")
    SEARCH_REAL_MIN, SEARCH_REAL_MAX = -np.inf, np.inf
    SEARCH_IMAG_MIN, SEARCH_IMAG_MAX = -np.inf, np.inf

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_complex_eigenvalues(filename):
    """Loads a 2-column text file (Real, Imag) into a 1D array of complex numbers."""
    try:
        data = np.loadtxt(filename, comments='#')
        return data[:, 0] + 1j * data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def load_complex_eigenvectors(real_file, imag_file):
    """
    Loads real and imaginary parts from text files and combines them into
    a 2D complex array. Assumes columns are eigenvectors and rows are components.
    """
    try:
        print(f"  Reading real parts from {real_file}...")
        real_data = np.loadtxt(real_file, comments='#')
        print(f"  Reading imaginary parts from {imag_file}...")
        imag_data = np.loadtxt(imag_file, comments='#')

        if real_data.shape != imag_data.shape:
            print(f"Error: Shape mismatch. Real part is {real_data.shape}, "
                  f"Imag part is {imag_data.shape}")
            return None

        return real_data + 1j * imag_data
    except Exception as e:
        print(f"Error loading files {real_file} and {imag_file}: {e}")
        return None

def compute_distance_matrix_fast(U, V):
    """
    Computes the complex Euclidean distance matrix between columns of U and V
    WITHOUT creating a large 3D array.

    U: shape (N, M1) (Test vectors)
    V: shape (N, M2) (Comparison vectors)
    Returns: shape (M1, M2)
    """

    print("\nCalculating deviation matrix...")
    U_sq = np.sum(np.real(U)**2 + np.imag(U)**2, axis=0) # (M1,)
    V_sq = np.sum(np.real(V)**2 + np.imag(V)**2, axis=0) # (M2,)

    # Guard against zero-norm vectors to avoid silent NaN in the MAC formula
    if np.any(U_sq == 0) or np.any(V_sq == 0):
        print("WARNING: One or more eigenvectors have zero norm. "
              "MAC values for those vectors will be set to NaN.")

    cross_term = np.dot(U.conj().T, V)

    # For phase independence, use the Modal Assurance Criterion
    # MAC = |u^H * v|^2 / (||u||^2 * ||v||^2)
    mac_matrix = (np.abs(cross_term)**2) / (U_sq[:, np.newaxis] * V_sq[np.newaxis, :])

    # We return (1 - MAC) so that 0.0 is a "perfect match" (acts like a distance)
    return 1.0 - mac_matrix

# ==============================================================================
# 3. MAIN LOGIC
# ==============================================================================
def main():
    print("\n--- Loading Data ---")
    start_time = time.time()

    print("\nLoading test case eigenvectors:")
    found_eigenvecs = load_complex_eigenvectors(TEST_REAL_FILE, TEST_IMAG_FILE)

    print("\nLoading comparison case eigenvectors:")
    compare_eigenvecs = load_complex_eigenvectors(COMPARE_REAL_FILE, COMPARE_IMAG_FILE)

    if found_eigenvecs is None or compare_eigenvecs is None:
        print("Failed to load one or both eigenvector sets. Exiting.")
        return

    n_comp_found, m_found = found_eigenvecs.shape
    n_comp_comp, m_comp = compare_eigenvecs.shape

    print(f"\nLoaded test case eigenvectors:   {m_found} vectors, {n_comp_found} components.")
    print(f"Loaded comparison case eigenvectors: {m_comp} vectors, {n_comp_comp} components.")

    # --- Filter the reference eigenvectors ---
    print("\n--- Filtering Comparison Case Data ---")
    compare_eigvals = load_complex_eigenvalues(COMPARE_EIGENVALS_FILE)

    if compare_eigvals is not None:
        if len(compare_eigvals) != m_comp:
            print(f"WARNING: The number of reference eigenvalues ({len(compare_eigvals)}) "
                  f"does not match the number of reference eigenvectors ({m_comp}). Filtering may be inaccurate.")

        real_parts = np.real(compare_eigvals)
        imag_parts = np.imag(compare_eigvals)

        # Create a boolean mask for the search box
        in_box_mask = (
            (real_parts >= SEARCH_REAL_MIN) & (real_parts <= SEARCH_REAL_MAX) &
            (imag_parts >= SEARCH_IMAG_MIN) & (imag_parts <= SEARCH_IMAG_MAX)
        )

        # Ensure mask length equals m_comp before boolean indexing.
        # If the eigenvalue file is shorter, pad with False (exclude unmatched columns).
        # If it is longer, truncate to m_comp.
        if len(in_box_mask) < m_comp:
            padding = np.zeros(m_comp - len(in_box_mask), dtype=bool)
            valid_mask = np.concatenate([in_box_mask, padding])
        else:
            valid_mask = in_box_mask[:m_comp]

        # Apply the mask to the columns
        compare_eigenvecs = compare_eigenvecs[:, valid_mask]
        n_comp_comp, m_comp = compare_eigenvecs.shape

        print(f"\nApplied search box filter: [{SEARCH_REAL_MIN}, {SEARCH_REAL_MAX}] x [{SEARCH_IMAG_MIN}, {SEARCH_IMAG_MAX}]")
        print(f"Filtered comparison case eigenvectors down to {m_comp} vectors.")
    else:
        print("Failed to load reference eigenvalues. Skipping filtering step.")

    if m_found == 0 or m_comp == 0:
        print("Not enough data to compute numerical differences. Exiting.")
        return

    if n_comp_found != n_comp_comp:
        print(f"\nWARNING: The number of components (rows) do not match! "
              f"({n_comp_found} vs {n_comp_comp}).")
        # Truncate to the smaller dimension to allow computation if testing with `head`
        min_comp = min(n_comp_found, n_comp_comp)
        found_eigenvecs = found_eigenvecs[:min_comp, :]
        compare_eigenvecs = compare_eigenvecs[:min_comp, :]
        print(f"Truncated both sets to {min_comp} components for valid comparison.")

    # --- Compute Distances ---
    print("\n--- Computing Deviations ---")
    dist_start = time.time()
    diff_matrix = compute_distance_matrix_fast(found_eigenvecs, compare_eigenvecs)
    print(f"Deviation matrix computed in {time.time() - dist_start:.2f} seconds.")

    # --- Compare Results ---
    print("\n--- Comparison Results ---")

    # diff_matrix contains (1.0 - MAC). Lower is better (0.0 = perfect match).
    # For each found eigenvector, find the minimum deviation to the reference set
    min_deviations = np.min(diff_matrix, axis=1)

    max_deviation = np.max(min_deviations)
    mean_deviation = np.mean(min_deviations)

    print(f"\nNumerical Difference Metrics (1.0 - MAC):")
    print(f"Note: 0.0 indicates a perfect phase/scale-invariant match, 1.0 indicates orthogonality.")
    print(f"  Maximum deviation (Worst match):  {max_deviation:.6e}")
    print(f"  Mean deviation (Average match):   {mean_deviation:.6e}")

    # For each found eigenvector, get the index of its closest reference match
    nearest_indices = np.argmin(diff_matrix, axis=1)

    # Check for duplicate matching (multiple found eigenvectors mapping to the same reference)
    unique_matches = len(np.unique(nearest_indices))

    if unique_matches < m_found:
        print(f"\nWARNING: {m_found - unique_matches} 'test' eigenvectors matched to the "
              f"SAME reference eigenvector. You likely have degenerate modes/repeated roots.")
    else:
        print("\nAll test eigenvectors map to unique reference eigenvectors.")

    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds.\n")

if __name__ == "__main__":
    main()
