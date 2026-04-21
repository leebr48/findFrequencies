# ====================================================================================
# USER INPUTS
# ====================================================================================

DATA_DIR = '.'  # Directory containing matrix files
BENDING_MATRIX_A = 'a_matrix.dat' # Name of the matrix A file (A x = lambda B x)
INERTIA_MATRIX_B = 'b_matrix.dat' # Name of the matrix B file (A x = lambda B x)

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

RUN_ERROR_CHECK      = True   # If True, compute and print ||Ax - lambda Bx||_2 / |lambda|.
SAVE_RESULTS         = True   # If True, save eigenvalues and eigenvectors to text files.
EIGENVAL_FILE        = 'found_eigenvalues.txt'
REAL_EIGENVEC_FILE   = 'found_eigenvectors_real.txt'
IMAG_EIGENVEC_FILE   = 'found_eigenvectors_imag.txt'
DIAG_FILE_STEM       = 'solver_diagnostics'  # All solver diagnostics are written here.
                                             # The file is always created; set to None to disable.

# Many of the functions below were adapted from https://github.com/jcmgray/quimb.

# ====================================================================================
# 0. IMPORTS
# ====================================================================================

import os
import sys
import datetime
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
    from the generalized eigenvalue problem A x = lambda B x,
    where lambda is the square of frequency and x describes the shear Alfven mode.

    Credit: Alexey Knyazev.
    """
    sim_dir:     str
    filename:    str
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

    # Diagnostics
    diag_file: str = 'solver_diagnostics.txt'  # Set to None or '' to disable.


# ====================================================================================
# 4. DIAGNOSTICS LOGGER
# ====================================================================================

class DiagnosticsLogger:
    """
    Writes structured solver diagnostics to a plain-text log file.

    Design
    ------
    * All *structured* text (section headers, key-value pairs, tables) is written
      through a Python file handle that is open exclusively on MPI rank 0.
    * PETSc / SLEPc object views (.view()) are appended by creating a short-lived
      PETSc ASCII viewer in APPEND mode, then immediately destroying it.  This
      keeps the two write paths from interleaving incorrectly.
    * The Python handle uses system-default block buffering (buffering=-1) rather
      than line buffering.  Explicit flushes are issued only at section boundaries
      and just before PETSc viewer calls, keeping I/O overhead negligible relative
      to the cost of the linear algebra.
    * On ranks > 0 all public methods are silent no-ops, so callers need not
      guard every call site.

    MPI collective discipline
    -------------------------
    Any method that internally calls a PETSc collective (mat.norm, mat.getInfo
    with GLOBAL_SUM, obj.view, etc.) must itself be called collectively on ALL
    ranks — not just rank 0 — even though only rank 0 writes anything to disk.
    Methods that are rank-0-only are clearly annotated below.

    EPS convergence monitoring
    --------------------------
    Call make_eps_monitor() to get a callback suitable for eps.setMonitor().
    The callback writes one line per Krylov iteration to the log file with no
    per-iteration flush, so it adds negligible overhead to the solve.
    """

    _WIDTH = 80  # character width for section banners

    def __init__(self, filepath: str, comm: MPI.Comm):
        self.filepath = filepath
        self.comm     = comm
        self.rank     = comm.Get_rank()
        self._fh: object = None   # Python file handle, rank 0 only

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Open the log file and write the file header.  Rank 0 only."""
        if self.rank != 0:
            return
        # Use system-default block buffering (-1) instead of line buffering (1).
        # This keeps per-iteration and per-eigenpair writes from causing OS-level
        # flushes on every newline, which would create a severe I/O bottleneck.
        self._fh = open(self.filepath, 'w', buffering=-1)
        w = self._WIDTH
        self._raw(f"{'='*w}")
        self._raw(f"  SOLVER DIAGNOSTICS LOG")
        self._raw(f"  Generated : {datetime.datetime.now().isoformat(timespec='seconds')}")
        self._raw(f"  MPI ranks : {self.comm.Get_size()}")
        try:
            self._raw(f"  PETSc ver : {'.'.join(str(v) for v in PETSc.Sys.getVersion())}")
            self._raw(f"  SLEPc ver : {'.'.join(str(v) for v in SLEPc.Sys.getVersion())}")
        except Exception:
            pass
        self._raw(f"{'='*w}\n")
        self._fh.flush()

    def close(self) -> None:
        """Write a closing footer and close the file.  Rank 0 only."""
        if self._fh is None:
            return
        w = self._WIDTH
        self._raw(f"\n{'='*w}")
        self._raw(f"  LOG CLOSED: {datetime.datetime.now().isoformat(timespec='seconds')}")
        self._raw(f"{'='*w}")
        self._fh.close()
        self._fh = None

    # ------------------------------------------------------------------
    # Structured text helpers — rank 0 only, not MPI collectives
    # ------------------------------------------------------------------

    def _raw(self, text: str) -> None:
        """Write a line unconditionally (internal use, rank 0 only)."""
        if self._fh is not None:
            self._fh.write(text + '\n')

    def section(self, title: str) -> None:
        """Print a clearly delimited section header and flush.  Rank 0 only."""
        if self._fh is None:
            return
        w = self._WIDTH
        self._raw(f"\n{'-'*w}")
        self._raw(f"  {title}")
        self._raw(f"{'-'*w}")
        # Flush at section boundaries so partial output is visible if the
        # process crashes mid-solve.
        self._fh.flush()

    def write(self, text: str) -> None:
        """Write an arbitrary line of text.  Rank 0 only."""
        self._raw(text)

    def kv(self, key: str, value) -> None:
        """Write a key-value pair with consistent alignment.  Rank 0 only."""
        self._raw(f"  {key:<42s} {value}")

    def blank(self) -> None:
        """Write a blank line.  Rank 0 only."""
        self._raw("")

    # ------------------------------------------------------------------
    # PETSc / SLEPc object views — MPI COLLECTIVE, call on all ranks
    # ------------------------------------------------------------------

    def view_petsc(self, obj, label: str = "",
                   fmt: 'PETSc.Viewer.Format' = None) -> None:
        """
        Append the PETSc/SLEPc .view() output to the log file.

        COLLECTIVE: must be called on all MPI ranks simultaneously.

        Parameters
        ----------
        obj   : any PETSc/SLEPc object that supports .view(viewer)
        label : optional annotation printed before the view block (rank 0 only)
        fmt   : PETSc viewer format.  Defaults to ASCII_INFO_DETAIL (verbose
                but does not dump every matrix entry).
        """
        if fmt is None:
            fmt = PETSc.Viewer.Format.ASCII_INFO

        # Flush the Python buffer and write the label on rank 0 before the
        # PETSc viewer opens the same file, to preserve correct ordering.
        if self._fh is not None:
            if label:
                self._raw(f"\n  [PETSc view: {label}]")
            self._fh.flush()

        # createASCII and obj.view() are collective on comm.
        viewer = PETSc.Viewer().createASCII(
            self.filepath,
            mode=PETSc.Viewer.Mode.APPEND,
            comm=self.comm,
        )
        viewer.pushFormat(fmt)
        try:
            obj.view(viewer)
        finally:
            viewer.popFormat()
            viewer.flush()
            viewer.destroy()

    def view_mat_info(self, mat: 'PETSc.Mat', label: str = "") -> None:
        """
        Log matrix summary statistics (shape, nnz, norms) followed by the
        PETSc ASCII_INFO block.

        COLLECTIVE: must be called on all MPI ranks simultaneously.

        mat.norm() and mat.getInfo(GLOBAL_SUM) are MPI collectives — they must
        execute on every rank regardless of whether that rank owns the log file.
        The results are computed collectively first; only rank 0 then writes them.
        """
        # --- Collective computations (all ranks participate) ---
        rows, cols = mat.getSize()                               # local read, same on all ranks
        info       = mat.getInfo(PETSc.Mat.InfoType.GLOBAL_SUM) # MPI collective
        frob       = mat.norm(PETSc.NormType.FROBENIUS)          # MPI collective
        inf_norm   = mat.norm(PETSc.NormType.INFINITY)           # MPI collective (max row sum)

        # --- Rank-0 write ---
        if self._fh is not None:
            self.section(f"Matrix info: {label}")
            self.kv("Global size (rows x cols)",        f"{rows} x {cols}")
            self.kv("Non-zeros allocated (global)",     int(info['nz_allocated']))
            self.kv("Non-zeros used (global)",          int(info['nz_used']))
            self.kv("Memory reported by PETSc (bytes)", f"{info['memory']:.3e}")
            self.kv("Mallocs during assembly",          int(info['mallocs']))
            self.kv("Frobenius norm",                   f"{frob:.6e}")
            self.kv("Infinity norm (max |row sum|)",    f"{inf_norm:.6e}")

        # PETSc ASCII_INFO block (collective).
        self.view_petsc(mat, fmt=PETSc.Viewer.Format.ASCII_INFO)

    # ------------------------------------------------------------------
    # EPS convergence monitor factory — rank 0 only inside callback
    # ------------------------------------------------------------------

    def make_eps_monitor(self, chunk_label: str = ""):
        """
        Return a callback suitable for eps.setMonitor().

        Each call to the callback (= one Krylov iteration) writes one line to
        the log showing: iteration number, converged count, and up to 3 leading
        error/eigenvalue estimates.

        Performance note: the callback does NOT flush the file handle on every
        call.  The block-buffered handle flushes automatically when its internal
        buffer fills, keeping per-iteration overhead essentially zero.
        Only rank 0 writes; the callback is safe to invoke from all ranks.
        """
        rank   = self.rank
        fh     = self._fh
        prefix = f"[{chunk_label}] " if chunk_label else ""

        def _monitor(eps, its, nconv, eigs, errors):
            if rank != 0 or fh is None:
                return
            n_show    = min(3, len(errors))
            err_parts = [f"|r[{k}]|={errors[k]:.2e}" for k in range(n_show)]
            eig_parts = [
                f"eig[{k}]={eigs[k].real:.4g}{eigs[k].imag:+.4g}j"
                for k in range(min(n_show, len(eigs)))
            ]
            err_str = "  ".join(err_parts) if err_parts else "(no estimates yet)"
            eig_str = "  ".join(eig_parts)
            line = (
                f"  {prefix}EPS iter {its:5d}:  nconv={nconv:4d}  {err_str}"
            )
            if eig_str:
                line += f"  |  {eig_str}"
            # No explicit flush here — block buffering keeps this cheap.
            fh.write(line + '\n')

        return _monitor

    # ------------------------------------------------------------------
    # Convergence-reason decoder — pure Python, no MPI
    # ------------------------------------------------------------------

    @staticmethod
    def converged_reason_str(reason) -> str:
        """
        Return a human-readable string for an SLEPc.EPS.ConvergedReason value.
        Falls back to the integer representation if the name is unknown.
        """
        try:
            name = reason.name   # works if reason is an IntEnum member
            return f"{name} ({int(reason)})"
        except AttributeError:
            pass
        _table = {
            1:  "CONVERGED_TOL",
            2:  "CONVERGED_USER",
           -1:  "DIVERGED_ITS",
           -2:  "DIVERGED_BREAKDOWN",
           -3:  "DIVERGED_SYMMETRY_LOST",
            0:  "CONVERGED_ITERATING",
        }
        v = int(reason)
        return _table.get(v, f"UNKNOWN({v})")


# ====================================================================================
# 5. SPARSE TO PETSC CONVERTER
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
# 6. EIGENSOLVER: KRYLOV-SCHUR OR CISS
# ====================================================================================

def _configure_linear_solver(
    st:            SLEPc.ST,
    linear_solver: str,
    logger:        DiagnosticsLogger = None,
) -> None:
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

    # Diagnostics: log the requested KSP/PC type before setUp.
    # The full factorization details appear in the post-setUp view below.
    if logger is not None:
        logger.kv("KSP type", "PREONLY  (direct solve, no iterations)")
        logger.kv("PC type",  "LU")
        logger.kv("Factor solver", linear_solver)


def solve_eigenproblem(
    A_sparse: sp.spmatrix,
    B_sparse: sp.spmatrix,
    comm:     MPI.Comm,
    config:   SolverConfig,
    logger:   DiagnosticsLogger = None,
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
    logger   : DiagnosticsLogger instance, or None to skip all diagnostics

    Returns
    -------
    On rank 0: list of (eigenvalue, error, vr_full, vi_full) tuples.
    On other ranks: empty list.
    """
    rank = comm.Get_rank()

    # ------------------------------------------------------------------
    # Log solver configuration (rank 0 only — no collectives here)
    # ------------------------------------------------------------------
    if logger is not None:
        logger.section("SOLVER CONFIGURATION")
        logger.kv("Algorithm",             config.method)
        logger.kv("Real search range",     f"[{config.real_min}, {config.real_max}]")
        logger.kv("Imaginary search range", f"[{config.imag_min}, {config.imag_max}]")
        logger.kv("Linear solver (ST)",    config.linear_solver)
        logger.kv("Problem type",          "GNHEP (generalized, non-Hermitian)")
        logger.kv("Hermitian hint set",    config.is_hermitian)
        logger.kv("Check residual errors", config.check_error)
        logger.blank()
        if config.method == 'krylovschur':
            logger.kv("KS num eigenvalues",      config.ks_num_eigenvals)
            logger.kv("KS subspace dim (ncv)",   config.ks_subspace_dim)
            logger.kv("KS search complex plane", config.ks_search_complex)
        else:
            logger.kv("CISS integration points", config.ciss_num_points)
            logger.kv("CISS blocksize",           config.ciss_blocksize)
            logger.kv("CISS moments",             config.ciss_moments)
            logger.kv("CISS SVD delta",           config.ciss_delta)
            logger.kv("CISS spurious threshold",  config.ciss_threshold)

    # ------------------------------------------------------------------
    # Convert matrices and log their statistics
    # ------------------------------------------------------------------
    pA = convert_sparse_to_petsc(A_sparse, comm=comm, is_hermitian=config.is_hermitian)
    pB = convert_sparse_to_petsc(B_sparse, comm=comm, is_hermitian=config.is_hermitian)

    # view_mat_info is COLLECTIVE (calls mat.norm, mat.getInfo internally).
    if logger is not None:
        logger.view_mat_info(pA, label="A  (LHS)")
        logger.view_mat_info(pB, label="B  (RHS)")

    # ------------------------------------------------------------------
    # Determine chunk intervals
    # ------------------------------------------------------------------
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

    if logger is not None:
        logger.section("CHUNK PLAN")
        logger.kv("Number of chunks", len(intervals))
        for k, (lo, hi) in enumerate(intervals):
            logger.kv(f"  Chunk {k+1}", f"real in [{lo:.4f}, {hi:.4f}]")

    # Allocate two persistent work vectors for eigenvector extraction.
    # createVecRight() returns only the right (column) vector, avoiding the
    # memory leak that results from discarding the left vector via createVecs().
    vr = pA.createVecRight()
    vi = pA.createVecRight()

    results = []

    # ------------------------------------------------------------------
    # Main solver loop (one iteration per chunk)
    # ------------------------------------------------------------------
    for chunk_idx, (c_min, c_max) in enumerate(intervals):
        comm.Barrier()

        chunk_label = (
            f"CHUNK {chunk_idx + 1}/{len(intervals)}: [{c_min:.2f}, {c_max:.2f}]"
            if len(intervals) > 1 else "SOLVE"
        )

        if rank == 0:
            chunk_tag = (
                f" (CHUNK {chunk_idx + 1}/{len(intervals)}: [{c_min:.2f}, {c_max:.2f}])"
                if len(intervals) > 1 else ""
            )
            print(f"\n--- CONFIGURING SOLVER{chunk_tag} ---", flush=True)

        if logger is not None:
            logger.section(f"SOLVER SETUP — {chunk_label}")

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
            _configure_linear_solver(st, config.linear_solver, logger=logger)

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
            _configure_linear_solver(st, config.linear_solver, logger=logger)

        else:
            raise ValueError(
                f"Unknown method '{config.method}'. Choose 'krylovschur' or 'ciss'."
            )

        eps.setFromOptions()

        # Attach the convergence monitor *before* setUp/solve so it captures
        # every Krylov iteration from the very first one.
        if logger is not None:
            eps.setMonitor(logger.make_eps_monitor(chunk_label=chunk_label))
            logger.section(f"EPS CONVERGENCE MONITOR — {chunk_label}")
            logger.write("  (one line per Krylov iteration)")

        if rank == 0:
            print("--- EXECUTING SETUP & SOLVE ---", flush=True)

        eps.setUp()
        eps.solve()

        # ------------------------------------------------------------------
        # Post-solve diagnostics: view fully configured solver objects.
        # All view_petsc calls below are COLLECTIVE.
        # ------------------------------------------------------------------
        if logger is not None:
            logger.section(f"POST-SETUP SOLVER VIEWS — {chunk_label}")

            # EPS: shows type, tolerances, dimensions, problem type, etc.
            logger.view_petsc(eps, label="EPS (eigenvalue solver)")

            # ST and its KSP/PC: shows factorization details, solver package, etc.
            st_inner  = eps.getST()
            logger.view_petsc(st_inner, label="ST (spectral transform)")

            ksp_inner = st_inner.getKSP()
            logger.view_petsc(ksp_inner, label="KSP (linear solver inside ST)")

            pc_inner  = ksp_inner.getPC()
            logger.view_petsc(pc_inner, label="PC (preconditioner / factorization)")

            # Converged reason and iteration count (local reads, not collective).
            reason     = eps.getConvergedReason()
            n_iters    = eps.getIterationNumber()
            reason_str = DiagnosticsLogger.converged_reason_str(reason)

            logger.section(f"CONVERGENCE REPORT — {chunk_label}")
            logger.kv("EPS converged reason",    reason_str)
            logger.kv("Total Krylov iterations", n_iters)

        # ------------------------------------------------------------------
        # Extract results
        # ------------------------------------------------------------------
        nconv = eps.getConverged()

        if rank == 0:
            print(f"Converged eigenpairs in this chunk: {nconv}", flush=True)

        if logger is not None:
            logger.kv("Converged eigenpairs this chunk", nconv)

        if nconv == 0:
            if logger is not None:
                logger.write(
                    "  WARNING: zero eigenpairs converged in this chunk. "
                    "Consider increasing KS_SUBSPACE_DIM, relaxing tolerances, "
                    "or checking for numerical breakdown in the converged reason above."
                )
            eps.destroy()
            continue

        if rank == 0 and config.check_error:
            print("--- COMPUTING RELATIVE ERRORS ---", flush=True)

        # ------------------------------------------------------------------
        # Per-eigenpair residual table
        # ------------------------------------------------------------------
        if logger is not None and config.check_error:
            logger.section(f"PER-EIGENPAIR RELATIVE RESIDUALS — {chunk_label}")
            logger.write(
                f"  {'idx':>5}  {'Re(lambda)':>18}  {'Im(lambda)':>18}  "
                f"{'||Ax-lBx||/|l|':>18}"
            )
            logger.write(f"  {'-'*5}  {'-'*18}  {'-'*18}  {'-'*18}")

        # --- Batched eigenvector extraction and MPI gathering ----------------
        # Accumulate all local vector fragments into (nconv, local_n) matrices,
        # then perform a single comm.gather per chunk instead of one per eigenpair.
        # This reduces MPI collective overhead from 2*nconv calls to 2 per chunk.
        chunk_vals   = []
        chunk_errors = []
        vr_locals    = []
        vi_locals    = []

        # Accumulate residual lines in a list and write them in one block after
        # the loop, rather than issuing one write per eigenpair inside the hot path.
        residual_lines = []

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

            if logger is not None and config.check_error and rank == 0:
                residual_lines.append(
                    f"  {i:>5d}  {val.real:>18.8e}  {val.imag:>18.8e}  "
                    f"{error:>18.6e}"
                )

        # Write the entire residual table in one shot.
        if residual_lines and logger is not None:
            logger.write('\n'.join(residual_lines))

        if logger is not None and config.check_error:
            errors_arr = np.array(chunk_errors)
            logger.blank()
            logger.kv("  Max relative residual (chunk)",  f"{errors_arr.max():.6e}")
            logger.kv("  Min relative residual (chunk)",  f"{errors_arr.min():.6e}")
            logger.kv("  Mean relative residual (chunk)", f"{errors_arr.mean():.6e}")

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

    # ------------------------------------------------------------------
    # Grand total summary
    # ------------------------------------------------------------------
    if logger is not None:
        logger.section("GRAND TOTAL SUMMARY")
        logger.kv("Total eigenpairs returned (all chunks)", len(results))
        if config.check_error and results:
            all_errors = np.array([r[1] for r in results if r[1] is not None])
            if all_errors.size > 0:
                logger.kv("Max relative residual (all chunks)",  f"{all_errors.max():.6e}")
                logger.kv("Min relative residual (all chunks)",  f"{all_errors.min():.6e}")
                logger.kv("Mean relative residual (all chunks)", f"{all_errors.mean():.6e}")
                worst_idx = int(np.argmax(all_errors))
                worst_ev  = results[worst_idx][0]
                logger.kv(
                    "Worst eigenpair (global idx, eigenvalue)",
                    f"idx={worst_idx}  "
                    f"lambda={worst_ev.real:.6f}{worst_ev.imag:+.6f}j"
                )

    return results


# ====================================================================================
# 7. DRIVER
# ====================================================================================

if __name__ == "__main__":
    comm = get_default_comm()
    rank = comm.Get_rank()

    # ------------------------------------------------------------------
    # Open diagnostics log
    # ------------------------------------------------------------------
    logger = None
    if DIAG_FILE_STEM:
        # Insert a timestamp into the filename so each run produces a fresh file
        # and old logs are never overwritten.  E.g. 'solver_diagnostics.txt'
        # becomes 'solver_diagnostics_20260408_143022.txt'.
        _ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        diag_path = f"{DIAG_FILE_STEM}_{_ts}.txt"

        logger = DiagnosticsLogger(filepath=diag_path, comm=comm)
        logger.open()

        # Log top-level user configuration for reproducibility.
        if rank == 0:
            logger.section("USER INPUT PARAMETERS")
            logger.kv("DATA_DIR",          DATA_DIR)
            logger.kv("BENDING_MATRIX_A",  BENDING_MATRIX_A)
            logger.kv("INERTIA_MATRIX_B",  INERTIA_MATRIX_B)
            logger.kv("SYMMETRIZE",        SYMMETRIZE)
            logger.kv("SEARCH_REAL_MIN",   SEARCH_REAL_MIN)
            logger.kv("SEARCH_REAL_MAX",   SEARCH_REAL_MAX)
            logger.kv("SEARCH_IMAG_MIN",   SEARCH_IMAG_MIN)
            logger.kv("SEARCH_IMAG_MAX",   SEARCH_IMAG_MAX)
            logger.kv("ALGO",              ALGO)
            logger.kv("LINEAR_SOLVER",     LINEAR_SOLVER)
            logger.kv("RUN_ERROR_CHECK",   RUN_ERROR_CHECK)
            logger.kv("SAVE_RESULTS",      SAVE_RESULTS)
            logger.kv("DIAG_FILE_STEM",         diag_path)
            if ALGO == 'krylovschur':
                logger.blank()
                logger.kv("KS_NUM_EIGENVALS",  KS_NUM_EIGENVALS)
                logger.kv("KS_SUBSPACE_DIM",   KS_SUBSPACE_DIM)
                logger.kv("KS_SEARCH_COMPLEX", KS_SEARCH_COMPLEX)
            else:
                logger.blank()
                logger.kv("CISS_NUM_POINTS", CISS_NUM_POINTS)
                logger.kv("CISS_BLOCKSIZE",  CISS_BLOCKSIZE)
                logger.kv("CISS_MOMENTS",    CISS_MOMENTS)
                logger.kv("CISS_DELTA",      CISS_DELTA)
                logger.kv("CISS_THRESHOLD",  CISS_THRESHOLD)

    # --- Load matrices -------------------------------------------------------
    A_sparse = SparseMatrixLoader(DATA_DIR, BENDING_MATRIX_A).matrix
    B_sparse = SparseMatrixLoader(DATA_DIR, INERTIA_MATRIX_B).matrix

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
        diag_file         = diag_path,
    )

    # --- Solve ---------------------------------------------------------------
    found_results = solve_eigenproblem(A_sparse, B_sparse, comm, config, logger=logger)

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

        if logger is not None:
            logger.section("POST-PROCESS: IN-RANGE FILTER")
            logger.kv("Total eigenpairs returned by solver", len(found_results))
            logger.kv("Eigenpairs within user-specified range", len(in_range))

        if in_range:
            if RUN_ERROR_CHECK:
                max_item = max(in_range, key=lambda x: x[1][1])
                max_idx, (max_ev, max_err, _, _) = max_item
                print(
                    f"\n[DIAGNOSTIC] Largest relative error in range: {max_err:.2e} "
                    f"at eigenvalue index {max_idx} "
                    f"(value: {max_ev.real:.6f} + {max_ev.imag:.6f}j)."
                )
                if logger is not None:
                    logger.kv(
                        "Largest in-range relative residual",
                        f"{max_err:.6e}  "
                        f"(idx={max_idx}, lambda={max_ev.real:.6f}{max_ev.imag:+.6f}j)"
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

                if logger is not None:
                    logger.section("OUTPUT FILES")
                    logger.kv("Eigenvalue file",            EIGENVAL_FILE)
                    logger.kv("Real eigenvector file",      REAL_EIGENVEC_FILE)
                    logger.kv("Imaginary eigenvector file", IMAG_EIGENVEC_FILE)
                    logger.kv("Eigenpairs written",         len(in_range))

    # --- Close diagnostics log -----------------------------------------------
    if logger is not None:
        logger.close()
        if rank == 0:
            print(f"\n[DIAGNOSTICS] Log written to: {diag_path}")
