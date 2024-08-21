# Quantum Teleportation Simulation

- [x] 1. Teleportation based on state vector of qubit
   
- [X] 2. Teleportation based on density matrix of qubit

- [X] 2. The influence of noise(like Dephasing) `visualization`

## Install
### Clone git
```
git clone https://github.com/ChuhanZhou/Quantum_Teleportation_Simulation.git
```
### Deploy virtual environment
```
#environment name: env_SCIQIS_test
conda env create -f environment.yml
```
## Demo
### Install Jupyter Lab
```
pip install jupyterlab
```
### Run Jupyter Lab
```
jupyter lab
```
### Install Jupyter Notebook
```
pip install notebook
```
### Run Jupyter Notebook
```
jupyter notebook
```
### Create Jupyter kernel base on virtual environment
```
conda activate env_SCIQIS_test
pip install ipython
pip install ipykernel
python -m ipykernel install --user --name=env_SCIQIS_test
```
### Check available kernels
```
ipython kernelspec list
```
### Open [demo.ipynb](/demo.ipynb) in Jupyter Lab, Jupyter Notebook or IDE