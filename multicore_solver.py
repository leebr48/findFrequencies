# This script was adapted from https://github.com/jcmgray/quimb.

import os
import sys
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
import warnings

# Silence the harmless slepc4py complex() deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================================================================================
# 1. MPI / PETSc / SLEPc Initialization 
# ====================================================================================
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import slepc4py
slepc4py.init(sys.argv)
from slepc4py import SLEPc

from mpi4py import MPI

def get_default_comm():
    return MPI.COMM_WORLD

# ====================================================================================
# 2. Data loader from Alexey Knyazev
# ====================================================================================
@dataclass
class FieldBendingMatrix:
    '''
    Field bending (sparse) matrix A from Az = lambda Bz generalized eigenvalue problem.
    Stored in a_matrix.dat output of AE3D code,
    where lambda is the square of frequency, and eigenvector z is the shear Alfven mode.
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
    where lambda is the square of frequency, and eigenvector z is the shear Alfven mode.
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

# ====================================================================================
# 3. Dense to sparse converter using PETSc machinery
# ====================================================================================
def convert_dense_to_petsc(dense_mat, comm=None):
    if comm is None: comm = get_default_comm()
    
    pmat = PETSc.Mat().create(comm=comm)
    pmat.setSizes(dense_mat.shape)
    pmat.setType(PETSc.Mat.Type.AIJ)
    pmat.setUp()

    rstart, rend = pmat.getOwnershipRange()
    for i in range(rstart, rend):
        row_data = dense_mat[i, :]
        nonzero_cols = np.nonzero(row_data)[0].astype(PETSc.IntType)
        nonzero_vals = row_data[nonzero_cols].astype(PETSc.ScalarType)
        if len(nonzero_cols) > 0:
            pmat.setValues(i, nonzero_cols, nonzero_vals)

    pmat.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    pmat.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    return pmat

# ====================================================================================
# 4. Solver # FIXME generalize to have different methods at some point
# ====================================================================================
def solve_eigenproblem(A_dense, B_dense, comm):
    rank = comm.Get_rank()
    
    pA = convert_dense_to_petsc(A_dense, comm=comm)
    pB = convert_dense_to_petsc(B_dense, comm=comm)

    comm.Barrier()
    if rank == 0: print(f"\n--- CONFIGURING SOLVER ---", flush=True)
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(pA, pB)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP) 
    
    # Configure CISS
    eps.setType(SLEPc.EPS.Type.CISS)
    
    # Keep MPI communicators intact and route solves through primary ST
    opts = PETSc.Options()
    opts.setValue('-eps_ciss_usest', 1)      
    
    real_min, real_max = 1.0e2, 5.0e3 # FIXME generalize to input
    imag_bound = 1.0e-4 # FIXME generalize to input
    rg = eps.getRG()
    rg.setType(SLEPc.RG.Type.INTERVAL)
    rg.setIntervalEndpoints(real_min, real_max, -imag_bound, imag_bound)
    
    # ----------------------------------------------------------------------
    # Use SUPERLU_DIST for parallel factorization # FIXME generalize
    # ----------------------------------------------------------------------
    st = eps.getST()
    
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY) 
    
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType(PETSc.Mat.SolverType.SUPERLU_DIST)
    # ----------------------------------------------------------------------

    eps.setFromOptions()

    comm.Barrier()
    if rank == 0: print(f"--- EXECUTING SETUP (LU FACTORIZATION) ---", flush=True)
    eps.setUp()

    comm.Barrier()
    if rank == 0: print(f"--- EXECUTING SOLVE (CONTOUR INTEGRATION) ---", flush=True)
    eps.solve()

    comm.Barrier()
    if rank == 0: print(f"--- EXTRACTING RESULTS ---", flush=True)
    nconv = eps.getConverged() # FIXME would be useful to know how many un-converged eigenpairs there are! 
    eigenvalues = []
    for i in range(nconv):
        val = eps.getEigenvalue(i)
        eigenvalues.append(val)

    eps.destroy()
    pA.destroy()
    pB.destroy()
    
    return eigenvalues

# ====================================================================================
# Driver script
# ====================================================================================
if __name__ == "__main__":
    comm = get_default_comm()
    rank = comm.Get_rank()
    
    data_dir = '.' #FIXME should be another option
    A_dense = FieldBendingMatrix(data_dir).load_matrix().toarray() # FIXME switching these back to sparse matrices (without breaking transfer into PETSC) would be best, if possible
    B_dense = InertiaMatrix(data_dir).load_matrix().toarray()
    
    found_evals = solve_eigenproblem(A_dense, B_dense, comm)
    
    if rank == 0:
        print(f"\n[SUCCESS] Found {len(found_evals)} eigenvalues inside the contour.", flush=True)
        for idx, ev in enumerate(found_evals):
            print(f"   Eigenvalue {idx}: {ev}", flush=True)
