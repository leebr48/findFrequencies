import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

import slepc4py
slepc4py.init(sys.argv)
from slepc4py import SLEPc

# 1. Check the globally configured scalar type
print(f"PETSc/SLEPc Scalar Type: {PETSc.ScalarType.__name__}")

# 2. Verify that PETSc accepts complex arithmetic
vec = PETSc.Vec().createSeq(1)
vec[0] = 1.0 + 2.0j
print(f"Test Vector Value:       {vec[0]}")
