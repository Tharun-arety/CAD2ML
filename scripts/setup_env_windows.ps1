# Native Windows development environment (no Docker required).
# Creates the conda env and applies the BLAS/OpenMP fix recorded in DECISIONS.md (D-008).
$ErrorActionPreference = "Stop"
mamba env create -f environment.yml -y
# conda-forge MKL BLAS crashed numpy matmul on this host (0xC06D007F) and the OpenMP build of OpenBLAS
# conflicts with the PyTorch wheel's libiomp5md.dll. Use the pthreads OpenBLAS build instead.
conda install -n cad2ml -y --no-deps -c conda-forge "libopenblas=0.3.34=pthreads_h952cde7_2"
conda run -n cad2ml python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
conda run -n cad2ml python scripts/spike_occt.py
