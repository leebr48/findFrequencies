# ====================================================================================
# USER INPUTS
# ====================================================================================

DATA_DIR = '.'  # Directory containing a_matrix.dat and b_matrix.dat

SYMMETRIZE = False  # If True, replace A <- 0.5*(A + A*^T) and B <- 0.5*(B + B*^T)

SEARCH_REAL_MIN = 100.0   # Lower real bound for eigenvalue search
SEARCH_REAL_MAX = 5000.0  # Upper real bound for eigenvalue search
SEARCH_IMAG_MIN = -20.0   # Lower imaginary bound for eigenvalue search
SEARCH_IMAG_MAX = 20.0    # Upper imaginary bound for eigenvalue search

ALGO = 'krylovschur'
# 'krylovschur' is fast and should be used for real eigenvalues (a "hammer").
# 'ciss' is much slower but searches a region in the complex plane (a "scalpel").

KS_NUM_EIGENVALS  = 1500   # (Krylov-Schur only) Number of eigenvalues to find.
                           # Make this large enough that some eigenvalues fall outside your range.
KS_SUBSPACE_DIM   = 3000   # (Krylov-Schur only) Krylov subspace dimension.
                           # More memory -> higher accuracy. 2 * KS_NUM_EIGENVALS works well.
KS_SEARCH_COMPLEX = False  # If True, Krylov-Schur searches a disk in the complex plane.

CISS_NUM_POINTS = 128   # (CISS only) Integration points. Raise if max error is high.
CISS_BLOCKSIZE  = 64    # (CISS only) blocksize * moments = max eigenvalues per chunk.
CISS_MOMENTS    = 4     # (CISS only) See above. Raise if you hit the ceiling (needs more RAM).
CISS_DELTA      = 1e-4  # (CISS only) SVD cutoff for eigenvalue filtration.
                        # Increase if duplicates appear; decrease if true eigenvalues are missed.
CISS_THRESHOLD  = 1e-6  # (CISS only) Tolerance for rejecting spurious eigenvalues.

LINEAR_SOLVER = 'superlu_dist'  # Direct solver for the spectral transform.
                                # 'superlu_dist' or 'mumps'. Choose based on your PETSc build.

RUN_ERROR_CHECK = True   # If True, compute max(||Ax - lambda Bx||_2 / |lambda|) and print it.
SAVE_RESULTS    = True   # If True, save eigenvalues and eigenvectors to text files.
EIGENVAL_FILE        = 'found_eigenvalues.txt'
REAL_EIGENVEC_FILE   = 'found_eigenvectors_real.txt'
IMAG_EIGENVEC_FILE   = 'found_eigenvectors_imag.txt'

# Many of the functions below were adapted from https://github.com/jcmgray/quimb.

# ====================================================================================
# 0. IMPORTS
# ====================================================================================

import os
import sys
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
import warnings

# Silence the harmless slepc4py complex() deprecation warning.
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================================================================================
# 1. MPI / PETSC / SLEPC INITIALIZATION
# ====================================================================================

import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import slepc4py
slepc4py.init(sys.argv)
from slepc4py import SLEPc

from mpi4py import MPI


def get_default_comm() -> MPI.Comm:
    return MPI.COMM_WORLD


# ====================================================================================
# 2. MATRIX LOADER
# ====================================================================================

@dataclass
class SparseMatrixLoader:
    """
    Loads a sparse matrix from an AE3D-format .dat file into a SciPy COO matrix.

    The file format is a 1-indexed COO text file with three columns: row, col, value.
    This class is used to load both the A (field bending) and B (inertia) matrices
    from the generalized eigenvalue problem A z = lambda B z,
    where lambda is the square of frequency and z is the shear Alfven mode.

    Credit: Alexey Knyazev.
    """
    sim_dir:     str
    filename:    str
    description: str          = ""
    file_path:   str          = field(init=False)
    matrix:      sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, self.filename)
        self.matrix = self._load()

    def _load(self) -> sp.coo_matrix:
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Matrix file not found: {self.file_path}")

        with open(self.file_path, 'r') as f:
            data = np.loadtxt(f, dtype=[('i', int), ('j', int), ('value', float)])

        rows   = data['i'] - 1  # Convert from 1-indexed to 0-indexed
        cols   = data['j'] - 1
        values = data['value']
        size   = max(np.max(rows), np.max(cols)) + 1

        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))


# ====================================================================================
# 3. SOLVER CONFIGURATION
# ====================================================================================

@dataclass
class SolverConfig:
    """
    All configuration parameters for solve_eigenproblem.

    Krylov-Schur parameters (ks_*) are only used when method='krylovschur'.
    CISS parameters (ciss_*) are only used when method='ciss'.
    """
    method:  str   = 'krylovschur'  # 'krylovschur' or 'ciss'
    real_min: float = 100.0          # Lower real bound for eigenvalue search
    real_max: float = 5000.0         # Upper real bound for eigenvalue search
    imag_min: float = -20.0          # Lower imaginary bound for eigenvalue search
    imag_max: float = 20.0           # Upper imaginary bound for eigenvalue search

    # Krylov-Schur settings
    ks_num_eigenvals:  int  = 1500
    ks_subspace_dim:   int  = 3000
    ks_search_complex: bool = False

    # CISS settings
    ciss_num_points: int   = 128
    ciss_blocksize:  int   = 64
    ciss_moments:    int   = 4
    ciss_delta:      float = 1e-4
    ciss_threshold:  float = 1e-6

    # Shared settings
    is_hermitian:  bool = False          # If True, tag matrices as Hermitian for solver hints
    linear_solver: str  = 'superlu_dist' # 'superlu_dist' or 'mumps'
    check_error:   bool = True           # Compute ||Ax - lambda Bx||_2 / |lambda| per eigenpair


# ====================================================================================
# 4. SPARSE TO PETSC CONVERTER
# ====================================================================================

def convert_sparse_to_petsc(
    sparse_mat:   sp.spmatrix,
    comm:         MPI.Comm = None,
    is_hermitian: bool     = False,
) -> PETSc.Mat:
    """
    Convert a SciPy sparse matrix to a distributed PETSc AIJ (CSR) matrix.

    Each MPI rank is assigned a contiguous block of rows. The CSR structure is
    used to preallocate exactly the right number of non-zeros per local row,
    which avoids repeated internal reallocation during assembly and is
    significantly faster than the default dynamic-allocation path.
    """
    if comm is None:
        comm = get_default_comm()

    csr     = sparse_mat.tocsr()
    indptr  = csr.indptr
    indices = csr.indices
    data    = csr.data

    pmat = PETSc.Mat().create(comm=comm)
    pmat.setSizes(csr.shape)
    pmat.setType(PETSc.Mat.Type.AIJ)

    if is_hermitian:
        pmat.setOption(PETSc.Mat.Option.HERMITIAN, True)

    pmat.setUp()
    rstart, rend = pmat.getOwnershipRange()

    # Preallocate using the exact per-row non-zero count for this rank's rows.
    # Passing the total nnz per row (not split by diagonal/off-diagonal block)
    # is a safe over-estimate; NEW_NONZERO_ALLOCATION_ERR=False suppresses the
    # resulting PETSc warning without any loss of correctness.
    local_nnz = np.diff(indptr[rstart:rend + 1]).astype(PETSc.IntType)
    pmat.setPreallocationNNZ(local_nnz)
    pmat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Insert non-zeros row by row for the rows owned by this rank.
    for i in range(rstart, rend):
        s = indptr[i]
        e = indptr[i + 1]
        if s < e:
            pmat.setValues(
                i,
                indices[s:e].astype(PETSc.IntType),
                data[s:e].astype(PETSc.ScalarType),
                addv=PETSc.InsertMode.INSERT_VALUES,
            )

    pmat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    pmat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)

    return pmat


# ====================================================================================
# 5. EIGENSOLVER: KRYLOV-SCHUR OR CISS
# ====================================================================================

def _configure_linear_solver(st: SLEPc.ST, linear_solver: str) -> None:
    """Configure the KSP/PC inside a spectral transform for sparse direct LU solves."""
    solver_map = {
        'superlu_dist': PETSc.Mat.SolverType.SUPERLU_DIST,
        'mumps':        PETSc.Mat.SolverType.MUMPS,
    }
    if linear_solver not in solver_map:
        raise ValueError(
            f"Unknown linear_solver '{linear_solver}'. Choose 'superlu_dist' or 'mumps'."
        )
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType(solver_map[linear_solver])


def solve_eigenproblem(
    A_sparse: sp.spmatrix,
    B_sparse: sp.spmatrix,
    comm:     MPI.Comm,
    config:   SolverConfig,
) -> list:
    """
    Solve the generalized eigenvalue problem A x = lambda B x using SLEPc.

    Supports two solver strategies, selected via config.method:
      - 'krylovschur': Krylov-Schur with shift-and-invert. Fast for real eigenvalues.
      - 'ciss':        Contour integral (CISS). Slower but searches the complex plane.

    For CISS with a wide real interval, the search region is automatically
    subdivided into roughly square chunks (in the complex plane) and solved
    sequentially to stay within CISS's capacity limits.

    Eigenvector gathering is batched: all local vectors for a chunk are
    collected into a single (nconv, local_n) array before the MPI gather,
    reducing the number of collective calls from 2*nconv to 2 per chunk.

    Parameters
    ----------
    A_sparse : SciPy sparse matrix (left-hand side)
    B_sparse : SciPy sparse matrix (right-hand side)
    comm     : MPI communicator
    config   : SolverConfig instance with all algorithm parameters

    Returns
    -------
    On rank 0: list of (eigenvalue, error, vr_full, vi_full) tuples.
    On other ranks: empty list.
    """
    rank = comm.Get_rank()

    pA = convert_sparse_to_petsc(A_sparse, comm=comm, is_hermitian=config.is_hermitian)
    pB = convert_sparse_to_petsc(B_sparse, comm=comm, is_hermitian=config.is_hermitian)

    # --- Determine chunk intervals -------------------------------------------
    # Krylov-Schur processes the entire range in one shot.
    # CISS benefits from chunking the real axis into roughly square subregions
    # in the complex plane (aspect ratio ~ 1), which keeps each chunk well-
    # conditioned for the contour integration.
    if config.method == 'ciss':
        imag_width  = config.imag_max - config.imag_min
        real_width  = config.real_max - config.real_min
        num_chunks  = max(1, int(np.ceil(real_width / imag_width)))
        chunk_edges = np.linspace(config.real_min, config.real_max, num_chunks + 1)
        intervals   = [(chunk_edges[k], chunk_edges[k + 1]) for k in range(num_chunks)]
    else:
        intervals = [(config.real_min, config.real_max)]

    # Allocate two persistent work vectors for eigenvector extraction.
    # createVecRight() returns only the right (column) vector, avoiding the
    # memory leak that results from discarding the left vector via createVecs().
    vr = pA.createVecRight()
    vi = pA.createVecRight()

    results = []

    # --- Main solver loop (one iteration per chunk) --------------------------
    for chunk_idx, (c_min, c_max) in enumerate(intervals):
        comm.Barrier()

        if rank == 0:
            chunk_tag = (
                f" (CHUNK {chunk_idx + 1}/{len(intervals)}: [{c_min:.2f}, {c_max:.2f}])"
                if len(intervals) > 1 else ""
            )
            print(f"\n--- CONFIGURING SOLVER{chunk_tag} ---", flush=True)

        # Create a fresh EPS solver for each chunk so SLEPc does not reuse
        # the previous factorization or convergence history.
        eps = SLEPc.EPS().create(comm=comm)
        eps.setOperators(pA, pB)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        if config.method == 'krylovschur':
            eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
            eps.setDimensions(nev=config.ks_num_eigenvals, ncv=config.ks_subspace_dim)

            if config.ks_search_complex:
                eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            else:
                eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

            # Use the chunk left edge as the shift target for shift-and-invert.
            eps.setTarget(c_min)

            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            st.setShift(c_min)
            _configure_linear_solver(st, config.linear_solver)

        elif config.method == 'ciss':
            eps.setType(SLEPc.EPS.Type.CISS)

            # CISS dimensions are controlled by blocksize and moments internally;
            # setting ncv explicitly would conflict with that logic.
            opts = PETSc.Options()
            opts.setValue('-eps_ciss_usest',              1)
            opts.setValue('-eps_ciss_integration_points', config.ciss_num_points)
            opts.setValue('-eps_ciss_blocksize',          config.ciss_blocksize)
            opts.setValue('-eps_ciss_moments',            config.ciss_moments)
            opts.setValue('-eps_ciss_delta',              config.ciss_delta)
            opts.setValue('-eps_ciss_spurious_threshold', config.ciss_threshold)

            rg = eps.getRG()
            rg.setType(SLEPc.RG.Type.INTERVAL)
            rg.setIntervalEndpoints(c_min, c_max, config.imag_min, config.imag_max)

            st = eps.getST()
            st.setType(SLEPc.ST.Type.SINVERT)
            _configure_linear_solver(st, config.linear_solver)

        else:
            raise ValueError(
                f"Unknown method '{config.method}'. Choose 'krylovschur' or 'ciss'."
            )

        eps.setFromOptions()

        if rank == 0:
            print("--- EXECUTING SETUP & SOLVE ---", flush=True)

        eps.setUp()
        eps.solve()

        nconv = eps.getConverged()

        if rank == 0:
            print(f"Converged eigenpairs in this chunk: {nconv}", flush=True)

        if nconv == 0:
            eps.destroy()
            continue

        if rank == 0 and config.check_error:
            print("--- COMPUTING RELATIVE ERRORS ---", flush=True)

        # --- Batched eigenvector extraction and MPI gathering ----------------
        # Accumulate all local vector fragments into (nconv, local_n) matrices,
        # then perform a single comm.gather per chunk instead of one per eigenpair.
        # This reduces MPI collective overhead from 2*nconv calls to 2 per chunk.
        chunk_vals   = []
        chunk_errors = []
        vr_locals    = []
        vi_locals    = []

        for i in range(nconv):
            val = eps.getEigenvalue(i)
            eps.getEigenvector(i, vr, vi)
            error = (
                eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
                if config.check_error else None
            )
            chunk_vals.append(val)
            chunk_errors.append(error)
            vr_locals.append(vr.getArray().copy())
            vi_locals.append(vi.getArray().copy())

        # Shape: (nconv, local_n) on each rank.
        vr_matrix = np.array(vr_locals)
        vi_matrix = np.array(vi_locals)

        # Single gather per chunk: rank 0 receives a list of (nconv, local_n_k) arrays.
        # Concatenating along axis=1 reconstructs the full distributed eigenvectors.
        vr_gathered = comm.gather(vr_matrix, root=0)
        vi_gathered = comm.gather(vi_matrix, root=0)

        if rank == 0:
            vr_full_matrix = np.concatenate(vr_gathered, axis=1)  # (nconv, N)
            vi_full_matrix = np.concatenate(vi_gathered, axis=1)

            for i in range(nconv):
                results.append((
                    chunk_vals[i],
                    chunk_errors[i],
                    vr_full_matrix[i],
                    vi_full_matrix[i],
                ))

        eps.destroy()

    pA.destroy()
    pB.destroy()
    vr.destroy()
    vi.destroy()

    return results


# ====================================================================================
# 6. DRIVER
# ====================================================================================

if __name__ == "__main__":
    comm = get_default_comm()
    rank = comm.Get_rank()

    # --- Load matrices -------------------------------------------------------
    A_sparse = SparseMatrixLoader(
        DATA_DIR, 'a_matrix.dat', description="Field bending matrix A"
    ).matrix
    B_sparse = SparseMatrixLoader(
        DATA_DIR, 'b_matrix.dat', description="Inertia matrix B"
    ).matrix

    if SYMMETRIZE:
        A_sparse = 0.5 * (A_sparse + A_sparse.conj().T)
        B_sparse = 0.5 * (B_sparse + B_sparse.conj().T)

    # --- Build solver config -------------------------------------------------
    config = SolverConfig(
        method            = ALGO,
        real_min          = SEARCH_REAL_MIN,
        real_max          = SEARCH_REAL_MAX,
        imag_min          = SEARCH_IMAG_MIN,
        imag_max          = SEARCH_IMAG_MAX,
        ks_num_eigenvals  = KS_NUM_EIGENVALS,
        ks_subspace_dim   = KS_SUBSPACE_DIM,
        ks_search_complex = KS_SEARCH_COMPLEX,
        ciss_num_points   = CISS_NUM_POINTS,
        ciss_blocksize    = CISS_BLOCKSIZE,
        ciss_moments      = CISS_MOMENTS,
        ciss_delta        = CISS_DELTA,
        ciss_threshold    = CISS_THRESHOLD,
        is_hermitian      = SYMMETRIZE,
        linear_solver     = LINEAR_SOLVER,
        check_error       = RUN_ERROR_CHECK,
    )

    # --- Solve ---------------------------------------------------------------
    found_results = solve_eigenproblem(A_sparse, B_sparse, comm, config)

    # --- Post-process on rank 0 only -----------------------------------------
    if rank == 0:
        in_range = [
            (idx, res)
            for idx, res in enumerate(found_results)
            if (config.real_min <= res[0].real <= config.real_max)
            and (config.imag_min <= res[0].imag <= config.imag_max)
        ]

        print(
            f"\n[SUCCESS] {len(found_results)} total eigenvalues found, "
            f"{len(in_range)} within the specified range."
        )

        if in_range:
            if RUN_ERROR_CHECK:
                max_item = max(in_range, key=lambda x: x[1][1])
                max_idx, (max_ev, max_err, _, _) = max_item
                print(
                    f"\n[DIAGNOSTIC] Largest relative error in range: {max_err:.2e} "
                    f"at eigenvalue index {max_idx} "
                    f"(value: {max_ev.real:.6f} + {max_ev.imag:.6f}j)."
                )

            if SAVE_RESULTS:
                print("\nSaving eigenvalues and eigenvectors to text files...")

                filtered_vals = []
                filtered_vr   = []
                filtered_vi   = []

                for _, (val, _err, vr_full, vi_full) in in_range:
                    filtered_vals.append(val)
                    evec = vr_full + 1j * vi_full
                    filtered_vr.append(np.real(evec).astype(float))
                    filtered_vi.append(np.imag(evec).astype(float))

                val_array = np.column_stack((
                    np.real(filtered_vals),
                    np.imag(filtered_vals),
                ))
                np.savetxt(
                    EIGENVAL_FILE, val_array,
                    header="Real Imaginary", comments='', fmt='%.16e',
                )

                # Transpose: rows = DOF index, columns = eigenvector index
                np.savetxt(REAL_EIGENVEC_FILE, np.array(filtered_vr).T, fmt='%.16e')
                np.savetxt(IMAG_EIGENVEC_FILE, np.array(filtered_vi).T, fmt='%.16e')

                print("[SUCCESS] Files saved successfully.")
