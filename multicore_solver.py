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

        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']
        size = max(np.max(rows), np.max(cols)) + 1

        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))


@dataclass
class InertiaMatrix:
    '''
    Inertia matrix B from Az = lambda Bz generalized eigenvalue problem.
    Stored in b_matrix.dat output of AE3D code,
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

        rows = data['i'] - 1
        cols = data['j'] - 1
        values = data['value']
        size = max(np.max(rows), np.max(cols)) + 1

        return sp.coo_matrix((values, (rows, cols)), shape=(size, size))

# ====================================================================================
# 3. DENSE TO SPARSE CONVERTER
# ====================================================================================
def convert_dense_to_petsc(dense_mat, name, comm=None):
    if comm is None: comm = get_default_comm()
    
    pmat = PETSc.Mat().create(comm=comm)
    pmat.setSizes(dense_mat.shape)
    pmat.setType(PETSc.Mat.Type.AIJ)
    
    # Tag as Hermitian for good measure, though GNHEP uses standard Euclidean inner products
    pmat.setOption(PETSc.Mat.Option.HERMITIAN, True)
    
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
# 4. SOLVER: KRYLOV-SCHUR OR CISS
# ====================================================================================
def solve_eigenproblem(A_dense, B_dense, comm, real_min=0.0, real_max=5000.0, method="krylovschur"):
    rank = comm.Get_rank()
    
    pA = convert_dense_to_petsc(A_dense, "Matrix A", comm=comm)
    pB = convert_dense_to_petsc(B_dense, "Matrix B", comm=comm)

    comm.Barrier()
    if rank == 0: print(f"\n--- CONFIGURING SOLVER ({method.upper()}) ---", flush=True)
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(pA, pB)
    
    if method == "ciss":
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP) 
        eps.setType(SLEPc.EPS.Type.CISS)
        
        opts = PETSc.Options()
        opts.setValue('-eps_ciss_usest', 1)      
        
        imag_bound = 1.0e-4
        rg = eps.getRG()
        rg.setType(SLEPc.RG.Type.INTERVAL)
        rg.setIntervalEndpoints(real_min, real_max, -imag_bound, imag_bound)
        
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY) 
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType(PETSc.Mat.SolverType.SUPERLU_DIST)
        
    elif method == "krylovschur":
        # GNHEP safely handles indefinite Matrix B
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        
        # Target real_min (100) and grab the closest 1500 eigenvalues
        eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
        eps.setTarget(real_min) 
        eps.setDimensions(nev=1500, ncv=3000) 
        
        st = eps.getST()
        st.setType(SLEPc.ST.Type.SINVERT)
        st.setShift(real_min)
        
        # Force LU via SuperLU_DIST (bypassing macOS MUMPS memory issues)
        ksp = st.getKSP()
        ksp.setType(PETSc.KSP.Type.PREONLY) 
        pc = ksp.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType(PETSc.Mat.SolverType.SUPERLU_DIST)
        
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'krylovschur' or 'ciss'.")
    
    eps.setFromOptions()

    comm.Barrier()
    if rank == 0: print(f"--- EXECUTING SETUP (FACTORIZATION) ---", flush=True)
    eps.setUp()

    comm.Barrier()
    if rank == 0: print(f"--- EXECUTING SOLVE ---", flush=True)
    eps.solve()

    comm.Barrier()
    if rank == 0: print(f"--- EXTRACTING RESULTS ---", flush=True)
    nconv = eps.getConverged() 
    
    if rank == 0: print(f"Converged Eigenpairs: {nconv}", flush=True)
    
    eigenvalues = []
    for i in range(nconv):
        val = eps.getEigenvalue(i)
        error = eps.computeError(i)  # Calculate the residual error for verification
        eigenvalues.append((val, error))

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
    
    A_sparse = FieldBendingMatrix(data_dir).load_matrix()
    B_sparse = InertiaMatrix(data_dir).load_matrix()
    
    A_sparse = 0.5 * (A_sparse + A_sparse.conj().T)
    B_sparse = 0.5 * (B_sparse + B_sparse.conj().T)
    
    A_dense = A_sparse.toarray() # FIXME switching these back to sparse matrices (without breaking transfer into PETSC) would be best, if possible
    B_dense = B_sparse.toarray()
    
    # Define your search boundaries here
    search_min = 100.0
    search_max = 5000.0
    
    # Toggle to 'ciss' to test Contour Integration later
    found_evals = solve_eigenproblem(A_dense, B_dense, comm, real_min=search_min, real_max=search_max, method="krylovschur")
    
    if rank == 0:
        # Filter the eigenvalues to only include those within the user-specified range
        in_range_evals = [(idx, ev, err) for idx, (ev, err) in enumerate(found_evals) if search_min <= ev.real <= search_max]
        
        print(f"\n[SUCCESS] {len(found_evals)} total eigenvalues found, {len(in_range_evals)} of which are in the user-specified range [{search_min}, {search_max}].")
        
        if in_range_evals:
            # 1. Print formatted columns for in-range eigenvalues
            for idx, ev, err in in_range_evals:
                # <4 aligns the index left, >15.6f right-aligns the float, >9.2e right-aligns the scientific notation
                print(f"   Eigenvalue {idx:<4}: {ev.real:>15.6f} + {ev.imag:>15.6f}j  |  Residual Error: {err:>9.2e}")
            
            # 2. Find and print the eigenvalue with the maximum residual error WITHIN the specified range
            max_item = max(in_range_evals, key=lambda x: x[2])
            max_idx, max_ev, max_err = max_item
            
            print(f"\n[DIAGNOSTIC] The largest relative error of the eigenvalues in this range is {max_err:.2e} for eigenvalue {max_idx} with value ({max_ev.real:.6f} + {max_ev.imag:.6f}j).")
