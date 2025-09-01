# SeisTools
This is a repo for earthquake detection, phase picking and association. To get started, you'll 
need 

- [Setup](#setup)
- [Documentation](#documentation)

## Setup
Setup can be done with 
- [docker](#docker)
- [devcontainers](#devcontainers)
- [local](#local-setup)

### Docker
Docker helps to simplify the setup process with some constraints of its own. Major cons are
- All the waveform data ***must*** be copied to this project directory to work with docker.
- There can be race conditions and the processes might get stuck when using fork vs. spawn in linux and mac
- You're sharing the cpu resources with your host system which can cause performance overhead

1. To get started, run `sh launch_container.sh` in terminal from the project directory. 
  - This opens the _bash_ program in docker, and mounts your current directory to the container. 
    The first time you run it, it might be slow and take a while to install dependencies. 
    Subsequent runs will be faster. It'll install all dependencies by default.
2. Run the script and specify the args correctly. In this example, I'm using waveforms in 
  `examples/data`. 
  ```python
  python seistools/pipelines/phase_picking.py -mdn munet -fp /eqphases/examples/data -frq 40 -fmax 19
  ```
3. After the program is run, the results and saved locally on your system in the `picks` directory. Type 
  `exit` in terminal to return to your local system.

### Devcontainers
Devcontainers also use docker, but with additional advantages. They can be opened in vscode with 
GUIs like a regular folder, or remotely on [github codespaces](https://github.com/features/codespaces). <br>
To launch it locally, open vscode and press `Cmd+Shift+P` to open the command palette. Search for 
`Dev Containers: Rebuild and Reopen in Container` and click it. The environment opens with all 
dependencies installed. <br>
Major con is that all the waveform data has to be inside the repo to run predictions. You can 
detect phases using ``python seistools/pipelines/phase_picking.py -mdn munet -fp ./examples/data -frq 40 -fmax 19`

> [!Note]
> I used a relative path that references the project's root directory. Pointing to external 
  directories will trigger errors.


### Local setup
This requires multiple tools to be installed. Click on each link to read install instructions 
for different platforms. Programs should be installed according to list order.
1. [docker desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
  - Should install docker-compose by default on mac and windows
2. [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) - Compact version 
  of Anaconda.
3. [poetry](https://python-poetry.org/docs/) - package manager that resolves conflicts better 
  than conda. Install pipx with conda `conda install pipx`, then poetry with pipx 
  `pipx install poetry`.
4. Navigate into the repo directory and install dependencies with `poetry install`

Once installed, you can run the phase picking workflow on any folder / directory connected to 
your local system. You can use **absolute paths** for files on a harddrive etc
```bash
poetry run mlpicks -mdn munet -fp /Users/ppersaud/eqphases/examples/data -frq 100
```


## Documentation
Get the argument details with `poetry run mlpicks --help` in the repo directory
```bash
>$ poetry run mlpicks --help
Usage: mlpicks [OPTIONS]

  Orchestrates seismic phase picking using machine learning models.

  Supports munet, eqtransformer, and phasenet models for automated P and S
  phase detection in seismic waveforms.

Options:
  -mdn, --model_name [munet|eqtransformer|phasenet]
                                  Pretrained ML picker name  [required]
  -fp, --file_path PATH           Path to input waveforms file / directory
                                  [required]
  -sdr, --save_dir PATH           Output directory to save results. Directory
                                  is created if it doesnt exist.
  -frq, --frequency FLOAT         Sampling frequency of input data in Hz. When
                                  using directories, same freq. is applied to
                                  all files within the directory.  [required]
  -ovlp, --overlap FLOAT RANGE    Percentage overlap between consecutive
                                  windows (0-1, default: 0.25)  [0<=x<=1;
                                  required]
  -fmin, --fmin FLOAT             Lower frequency for bandpass filter in Hz
                                  (default: 1)  [required]
  -fmax, --fmax FLOAT             Upper frequency for bandpass filter in Hz
                                  (default: 20)  [required]
  -mps, --min_pick_sep INTEGER    Minimum separation distance between
                                  consecutive picks in seconds (default: 3)
                                  [required]
  -pt, --p_thresh FLOAT RANGE     Probability threshold for P-phase detection
                                  (default: 0.5)  [0<=x<=1]
  -st, --s_thresh FLOAT RANGE     Probability threshold for S-phase detection
                                  (default: 0.5)  [0<=x<=1]
  -v, --verbose                   Enable verbose logging
  --help                          Show this message and exit.
```
<br>

When run in terminal, all the files in the directory are processed by default in parallel. 
Predictions can also be run for single files by passing the file path to `-fp` argument.

> [!Note] 
> Folder search is not recursive, so you need to specify the correct directory for the scripts to work.

An example file for running the code in an interactive [notebook](/examples/munet.ipynb).