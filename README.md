# SeisTools
This is a repo for earthquake detection, phase picking and association. To get started, you'll 
need 

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

An example command with default arguments is 
`poetry run mlpicks -mdn munet -fp /Users/ppersaud/Research/eqphases/examples/data -frq 100`

> [!Note] 
> Folder search is not recursive, so you need to specify the correct directory for the scripts to work.