import os
import sys
import numpy as np
import warnings

# Silence the harmless slepc4py complex() deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ====================================================================================
# 1. MPI / PETSc / SLEPc INITIALIZATION
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
# 2. BULLETPROOF DATA LOADER
# ====================================================================================
def load_raw_coordinates(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")

    data = np.loadtxt(file_path, dtype=[('i', int), ('j', int), ('value', float)])
    rows, cols, values = data['i'] - 1, data['j'] - 1, data['value']
    max_size = max(np.max(rows), np.max(cols)) + 1
    return rows, cols, values, max_size

# ====================================================================================
# 3. DENSE TO SPARSE CONVERTER
# ====================================================================================
def convert_dense_to_petsc(dense_mat, name, comm=None):
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
# 4. SOLVER: CISS WITH SUPERLU_DIST
# ====================================================================================
def solve_eigenproblem(A_dense, B_dense, comm):
    rank = comm.Get_rank()
    
    pA = convert_dense_to_petsc(A_dense, "Matrix A", comm=comm)
    pB = convert_dense_to_petsc(B_dense, "Matrix B", comm=comm)

    comm.Barrier()
    if rank == 0: print(f"\n--- CONFIGURING SOLVER ---", flush=True)
    eps = SLEPc.EPS().create(comm=comm)
    eps.setOperators(pA, pB)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP) 
    
    # Configure CISS
    eps.setType(SLEPc.EPS.Type.CISS)
    
    # Keep MPI communicators intact and route solves through primary ST
    opts = PETSc.Options()
    opts.setValue('-eps_ciss_partitions', 1) 
    opts.setValue('-eps_ciss_usest', 1)      
    
    real_min, real_max = 1.0e2, 5.0e3
    imag_bound = 1.0e-4
    rg = eps.getRG()
    rg.setType(SLEPc.RG.Type.INTERVAL)
    rg.setIntervalEndpoints(real_min, real_max, -imag_bound, imag_bound)
    
    # ----------------------------------------------------------------------
    # USE SUPERLU_DIST FOR ROBUST PARALLEL FACTORIZATION
    # ----------------------------------------------------------------------
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    
    ksp = st.getKSP()
    ksp.setType(PETSc.KSP.Type.PREONLY) 
    
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    # The magic line: Swap MUMPS for SuperLU_DIST
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
    nconv = eps.getConverged()
    eigenvalues = []
    for i in range(nconv):
        val = eps.getEigenvalue(i)
        eigenvalues.append(val)

    eps.destroy()
    pA.destroy()
    pB.destroy()
    
    return eigenvalues

# ====================================================================================
# MAIN SCRIPT EXECUTION
# ====================================================================================
if __name__ == "__main__":
    comm = get_default_comm()
    rank = comm.Get_rank()
    
    data_dir = '.'
    file_a = os.path.join(data_dir, 'a_matrix.dat')
    file_b = os.path.join(data_dir, 'b_matrix.dat')

    rows_A, cols_A, vals_A, size_A = load_raw_coordinates(file_a)
    rows_B, cols_B, vals_B, size_B = load_raw_coordinates(file_b)
    
    global_size = max(size_A, size_B)
    
    if rank == 0:
        print(f"Loading matrices. Enforcing global size: {global_size} x {global_size}", flush=True)
        
    A_dense = np.zeros((global_size, global_size), dtype=np.complex128)
    B_dense = np.zeros((global_size, global_size), dtype=np.complex128)
    np.add.at(A_dense, (rows_A, cols_A), vals_A)
    np.add.at(B_dense, (rows_B, cols_B), vals_B)
    
    found_evals = solve_eigenproblem(A_dense, B_dense, comm)
    
    if rank == 0:
        print(f"\n[SUCCESS] Found {len(found_evals)} eigenvalues inside the contour.", flush=True)
        for idx, ev in enumerate(found_evals):
            print(f"   Eigenvalue {idx}: {ev}", flush=True)
