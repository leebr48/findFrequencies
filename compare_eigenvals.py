# ==============================================================================
# 0. USER INPUTS
# ==============================================================================
TEST_FILE = 'found_eigenvalues.txt'
COMPARE_FILE = 'asym_eigenvalues.txt'

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import numpy as np
from multicore_solver import SEARCH_REAL_MIN, SEARCH_REAL_MAX, SEARCH_IMAG_MIN, SEARCH_IMAG_MAX

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def load_complex_eigenvalues(filename):
    """Loads a 2-column text file (Real, Imag) into a 1D array of complex numbers."""
    try:
        # skiprows=1 skips the header if it exists
        data = np.loadtxt(filename, skiprows=1)
        return data[:, 0] + 1j * data[:, 1]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# ==============================================================================
# 3. MAIN LOGIC
# ==============================================================================
def main():
    print("--- Loading Data ---")
    found_eigvals = load_complex_eigenvalues(TEST_FILE)
    compare_eigvals = load_complex_eigenvalues(COMPARE_FILE)

    if found_eigvals is None or compare_eigvals is None:
        print("Failed to load one or both eigenvalue files. Exiting.")
        return

    print(f"Loaded {len(found_eigvals)} eigenvalues from {TEST_FILE}")
    print(f"Loaded {len(compare_eigvals)} eigenvalues from {COMPARE_FILE}")

    # --- 4. Filter the reference eigenvalues ---
    print("\n--- Filtering Reference Data ---")
    real_parts = np.real(compare_eigvals)
    imag_parts = np.imag(compare_eigvals)

    # Create a boolean mask for the search box
    in_box_mask = (
        (real_parts >= SEARCH_REAL_MIN) & (real_parts <= SEARCH_REAL_MAX) &
        (imag_parts >= SEARCH_IMAG_MIN) & (imag_parts <= SEARCH_IMAG_MAX)
    )
    
    filtered_compare = compare_eigvals[in_box_mask]
    print(f"Found {len(filtered_compare)} eigenvalues inside the search box bounds:")
    print(f"  Real: [{SEARCH_REAL_MIN}, {SEARCH_REAL_MAX}]")
    print(f"  Imag: [{SEARCH_IMAG_MIN}, {SEARCH_IMAG_MAX}]")

    # --- 5. Compare the two sets ---
    print("\n--- Comparison Results ---")
    
    n_found = len(found_eigvals)
    n_compare = len(filtered_compare)
    
    if n_found != n_compare:
        print(f"WARNING: Mismatched counts! The new solver found {n_found}, "
              f"but there are {n_compare} reference eigenvalues in the box.")
    else:
        print("SUCCESS: Both sets have the same number of eigenvalues in this region.")

    if n_found == 0 or n_compare == 0:
        print("Not enough data to compute numerical differences.")
        return

    # Calculate a full distance matrix between the two sets in the complex plane
    # Using NumPy broadcasting: (n_found, 1) - (1, n_compare) -> (n_found, n_compare)
    diff_matrix = np.abs(found_eigvals[:, np.newaxis] - filtered_compare[np.newaxis, :])

    # For each found eigenvalue, find the distance to its nearest neighbor in the filtered set
    min_distances = np.min(diff_matrix, axis=1)
    
    max_error = np.max(min_distances)
    mean_error = np.mean(min_distances)
    
    print(f"\nNumerical Difference Metrics (distance in complex plane):")
    print(f"  Maximum discrepancy: {max_error:.6e}")
    print(f"  Mean discrepancy:    {mean_error:.6e}")

    # Check for duplicate matching (e.g., if the solver missed a distinct eigenvalue 
    # and instead found the same eigenvalue twice)
    nearest_indices = np.argmin(diff_matrix, axis=1)
    unique_matches = len(np.unique(nearest_indices))
    
    if unique_matches < n_found:
        print(f"\nWARNING: {n_found - unique_matches} 'found' eigenvalues matched to the "
              f"SAME reference eigenvalue. You may have duplicate roots or missed modes.")
    else:
        print("\nAll found eigenvalues map to unique reference eigenvalues.")

if __name__ == "__main__":
    main()
