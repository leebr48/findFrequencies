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
def solve_eigenproblem(A_dense, B_dense, comm, real_min=0.0, real_max=5000.0, method="krylovschur", check_error=True):
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
    
    # Initialize PETSc Vectors to hold the eigenvectors temporarily
    vr, _ = pA.createVecs()
    vi = vr.duplicate()
    
    results = []
    for i in range(nconv):
        # Extract Eigenvalue and Eigenvector
        val = eps.getEigenvalue(i)
        eps.getEigenvector(i, vr, vi)
        
        # Calculate error only if toggled on
        error = eps.computeError(i) if check_error else None
        
        # Safely gather the distributed vector arrays across the MPI communicators
        vr_local = vr.getArray()
        vi_local = vi.getArray()
        
        vr_gathered = comm.gather(vr_local, root=0)
        vi_gathered = comm.gather(vi_local, root=0)
        
        if rank == 0:
            # Stitch the distributed arrays back together seamlessly
            vr_full = np.concatenate(vr_gathered)
            vi_full = np.concatenate(vi_gathered)
            results.append((val, error, vr_full, vi_full))
        else:
            results.append((val, error, None, None))

    eps.destroy()
    pA.destroy()
    pB.destroy()
    vr.destroy()
    vi.destroy()
    
    return results

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
    
    # Toggle check_error=True to calculate residuals
    run_error_check = True 
    
    found_results = solve_eigenproblem(A_dense, B_dense, comm, real_min=search_min, real_max=search_max, method="krylovschur", check_error=run_error_check)
    
    if rank == 0:
        # Filter the results to only include those within the user-specified range
        in_range_results = [(idx, res) for idx, res in enumerate(found_results) if search_min <= res[0].real <= search_max]
        
        print(f"\n[SUCCESS] {len(found_results)} total eigenvalues found, {len(in_range_results)} of which are in the user-specified range [{search_min}, {search_max}].")
        
        if in_range_results:
            if run_error_check:
                # Find and print the eigenvalue with the maximum residual error WITHIN the specified range
                max_item = max(in_range_results, key=lambda x: x[1][1])
                max_idx, (max_ev, max_err, _, _) = max_item
                print(f"\n[DIAGNOSTIC] The largest relative error of the eigenvalues in this range is {max_err:.2e} for eigenvalue {max_idx} with value ({max_ev.real:.6f} + {max_ev.imag:.6f}j).")
            
            print("\nSaving eigenvalues and eigenvectors to text files...")
            
            filtered_vals = []
            filtered_vr = []
            filtered_vi = []
            
            for idx, res in in_range_results:
                val, err, vr_full, vi_full = res
                
                filtered_vals.append(val)
                
                # SLEPc complex builds store the entire complex vector in `vr_full` and zero in `vi_full`.
                # Real builds split them. Adding them this way handles BOTH builds perfectly.
                evec_complex = vr_full + 1j * vi_full
                
                # Extract strict float arrays so NumPy savetxt doesn't stringify them
                filtered_vr.append(np.real(evec_complex).astype(float))
                filtered_vi.append(np.imag(evec_complex).astype(float))
            
            # Save Eigenvalues (Format: "# Real Imaginary")
            val_array = np.column_stack((np.real(filtered_vals), np.imag(filtered_vals)))
            np.savetxt("found_eigenvalues.txt", val_array, header="# Real Imaginary", comments='', fmt='%.16e')
            
            # Convert to 2D numpy arrays. Shape: (num_eigenvectors, matrix_size)
            vr_array = np.array(filtered_vr)
            vi_array = np.array(filtered_vi)
            
            # Save Transposed Arrays (.T makes it matrix_size rows by num_eigenvectors columns)
            # The format string '%.16e' guarantees clean, separated columns in scientific notation.
            np.savetxt("found_eigenvectors_real.txt", vr_array.T, fmt='%.16e')
            np.savetxt("found_eigenvectors_imag.txt", vi_array.T, fmt='%.16e')
            
            print("[SUCCESS] Files saved successfully to working directory.")
