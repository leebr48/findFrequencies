import os
import sys
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp

# ====================================================================================
# Quirk: Initialize PETSc and SLEPc BEFORE importing the modules, or syntax will break
# ====================================================================================
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import slepc4py
slepc4py.init(sys.argv)
from slepc4py import SLEPc
# ======================================================================

@dataclass
class FieldBendingMatrix:
    '''
    Field bending (sparse) matrix A from Az = lambda Bz generalized eigenvalue problem.
    Stored in a_matrix.dat output of AE3D code,
    where lambda is the square of frequency, and eigenvector z describes the shear Alfven mode.
    '''
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Field bending (sparse) matrix A from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, 'a_matrix.dat')
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, 'r') as file:
            data = np.loadtxt(file, dtype=[('i', int), ('j', int), ('value', float)])

        # Adjust indices for 0-based indexing in Python (if original indices are 1-based)
        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']

        # Find the maximum index for matrix dimension
        size = max(np.max(rows), np.max(cols)) + 1

        # Create the COO sparse matrix
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))

@dataclass
class InertiaMatrix:
    '''
    Inertia (sparse) matrix B from Az = lambda Bz generalized eigenvalue problem.
    Stored in a_matrix.dat output of AE3D code,
    where lambda is the square of frequency, and eigenvector z describes the shear Alfven mode.
    '''
    sim_dir: str
    file_path: str = field(init=False)
    matrix_description: str = field(default_factory=lambda: "Inertia matrix B from Az = lambda Bz generalized eigenvalue problem")
    matrix: sp.coo_matrix = field(init=False)

    def __post_init__(self):
        self.file_path = os.path.join(self.sim_dir, 'b_matrix.dat')
        self.matrix = self.load_matrix()

    def load_matrix(self):
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        with open(self.file_path, 'r') as file:
            data = np.loadtxt(file, dtype=[('i', int), ('j', int), ('value', float)])

        # Adjust indices for 0-based indexing in Python (original indices are 1-based)
        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']

        # Find the maximum index for matrix dimension
        size = max(np.max(rows), np.max(cols)) + 1

        # Create the COO sparse matrix
        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))

def scipy_coo_to_petsc(mat_coo):
    """
    Converts a scipy.sparse.coo_matrix into a distributed petsc4py.PETSc.Mat.
    """
    mat_csr = mat_coo.tocsr()

    petsc_mat = PETSc.Mat().create(comm=PETSc.COMM_WORLD)
    petsc_mat.setSizes(mat_csr.shape)
    petsc_mat.setType(PETSc.Mat.Type.AIJ)
    petsc_mat.setUp()

    Istart, Iend = petsc_mat.getOwnershipRange()

    # Preallocation (Using PETSc's native IntType to prevent Segfaults)
    d_nnz = np.zeros(Iend - Istart, dtype=PETSc.IntType)
    o_nnz = np.zeros(Iend - Istart, dtype=PETSc.IntType)

    for idx, i in enumerate(range(Istart, Iend)):
        row_start = mat_csr.indptr[i]
        row_end = mat_csr.indptr[i+1]
        cols = mat_csr.indices[row_start:row_end]

        d_nnz[idx] = np.sum((cols >= Istart) & (cols < Iend))
        o_nnz[idx] = len(cols) - d_nnz[idx]

    petsc_mat.setPreallocationNNZ((d_nnz, o_nnz))

    # Allow SLEPc to safely allocate new non-zeros when it computes (A - shift * B)
    petsc_mat.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # Insertion
    for i in range(Istart, Iend):
        row_start = mat_csr.indptr[i]
        row_end = mat_csr.indptr[i+1]

        # Force NumPy to cast to PETSc's compiled memory types
        cols = mat_csr.indices[row_start:row_end].astype(PETSc.IntType)
        vals = mat_csr.data[row_start:row_end].astype(PETSc.ScalarType)

        petsc_mat.setValues(i, cols, vals)

    petsc_mat.assemblyBegin()
    petsc_mat.assemblyEnd()

    return petsc_mat


def solve_generalized_eigenproblem(A_petsc, B_petsc, real_min, real_max, imag_bound=1.0, method='ciss', nev_guess=50):
    """
    Solves A x = lambda B x using SLEPc.
    """
    # Fix MUMPS configuration to prevent Mac/MPI Segfaults
    opts = PETSc.Options()
    opts['mat_mumps_icntl_14'] = 500  # Give MUMPS 500% more working memory than it estimates
    opts['mat_mumps_icntl_24'] = 1    # Enable Null pivot detection (crucial if B matrix is singular)

    eps = SLEPc.EPS().create()
    eps.setOperators(A_petsc, B_petsc)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setTolerances(tol=1e-8, max_it=1000)

    if method == 'ciss':
        eps.setType(SLEPc.EPS.Type.CISS)
        rg = eps.getRG()
        rg.setType(SLEPc.RG.Type.INTERVAL)
        rg.setIntervalEndpoints(real_min, real_max, -imag_bound, imag_bound)

        # Assign MUMPS to every internal KSP that CISS uses.
        ksps = eps.getCISSKSPs() 
        for ksp in ksps:
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

    elif method == 'krylov':
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setDimensions(nev=nev_guess)

        target_shift = (real_min + real_max) / 2.0
        eps.setTarget(target_shift)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(target_shift)

        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType(PETSc.Mat.SolverType.MUMPS)

    elif method == 'jd':
        # Jacobi-Davidson: Suggested by AE3D
        eps.setType(SLEPc.EPS.Type.JD)
        eps.setDimensions(nev=nev_guess)

        target_shift = (real_min + real_max) / 2.0
        eps.setTarget(target_shift)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

        st = eps.getST()
        st.setType(SLEPc.ST.Type.PRECOND)

        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.BCGS)  # BiCGSTAB iterative solver
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.BJACOBI)

    else:
        raise ValueError("Method must be 'ciss' or 'krylov'")

    PETSc.Sys.Print(f"Solving with {method.upper()}...")
    eps.solve()

    nconv = eps.getConverged()
    PETSc.Sys.Print(f"Number of converged eigenpairs: {nconv}")

    eigenvalues = []
    for i in range(nconv):
        val = eps.getEigenpair(i, None)
        if real_min <= val.real <= real_max:
            eigenvalues.append(val)

    return eigenvalues

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    
    # 1. Load your matrices using SciPy
    PETSc.Sys.Print("Loading SciPy matrices...")
    A_scipy = FieldBendingMatrix('.').load_matrix()
    B_scipy = InertiaMatrix('.').load_matrix()
    
    # 2. Convert to PETSc format with MPI preallocation
    PETSc.Sys.Print("Converting to PETSc Mat format...")
    A_petsc = scipy_coo_to_petsc(A_scipy)
    B_petsc = scipy_coo_to_petsc(B_scipy)
    
    # 3. Define your search bounds
    real_min, real_max = 0, 1.0e5
    imag_bound = 1.0e5
    
    # 4. Run the solver
    found_evals = solve_generalized_eigenproblem(
        A_petsc, B_petsc, 
        real_min=real_min, 
        real_max=real_max, 
        imag_bound=imag_bound,
        method='jd' 
    )
    
    PETSc.Sys.Print(f"Extraction complete. Found {len(found_evals)} eigenvalues strictly within the real bounds.")
