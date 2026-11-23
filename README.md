# AI631 Project: Quantum SVMs for RC Column Capacity Checks under Biaxial Bending

Rohan Imtiaz (25280058), MS AI, LUMS

The question is whether a fidelity-kernel QSVM on three qubits (inputs Pu, ex, ey) can learn the SAFE / UNSAFE boundary of a
reinforced concrete section under biaxial bending, and how noise distorts that boundary. The hypothesis is rejected if the QSVM's
boundary-band error is not lower than a tuned RBF SVM's at the same training size, with the same tuning effort, over five seeds with 95 percent CIs.

**Outcome.** The hypothesis is rejected for both feature maps on all three sections. Pooled over training sizes on the main section,
the paired difference in boundary-band error (QSVM minus RBF SVM) is -0.013 [-0.042, +0.016] for the angle map and -0.012 [-0.038, +0.015]
for the ZZ map. Under noise, the ZZ kernel (tuned to a very small bandwidth) is the more fragile one: its false SAFE rate rises from
3 percent to between 11 and 20 percent. On ibm_fez its kernel shrinks by lambda = 0.90.

## Folder layout

```
Project/
├── README.md
├── requirements.txt
├── data_generation.ipynb                         generates the main section dataset
├── 25280058_AI631_Project_Dataset.csv            pu_kN, ex_mm, ey_mm, status (3 ft x 2 ft section)
├── 25280058_AI631_Project_Dataset_Extended.csv   same rows with the intermediate capacity values
├── kaggle_runner.ipynb                           runs notebooks 00 to 06 on Kaggle CPU and zips the outputs
├── kaggle_runner_noise_frozen.ipynb              runs only 05_noise_hardware/01b and 06 on Kaggle CPU
│
├── qsvm_rc/                                      shared helper package (imported by every notebook)
│   ├── config.py           paths, seed, sizes, tuning grids, section definitions, IBM account name
│   ├── rc_capacity.py      strain compatibility + Bresler (band definition, plots, extra sections)
│   ├── data_utils.py       loading, labels, r = Pu / phi Pni, fixed splits, angle scaler
│   ├── metrics.py          F1, precision, recall, balanced accuracy, MCC, false SAFE rate, paired statistics
│   ├── quantum_kernels.py  exact NumPy kernels, Qiskit feature maps, sampled / noisy kernels, circuit resources
│   ├── models.py           classical baselines, QSVM tuning, learning-curve driver
│   └── plots.py            learning curves, predicted SAFE regions
│
├── 00_setup/
│   └── 00_environment_and_sanity_checks.ipynb
├── 01_data/
│   ├── 01_data_audit_and_fixed_splits.ipynb
│   └── 02_extra_sections_generation.ipynb
├── 02_classical/
│   └── 01_classical_baselines.ipynb
├── 03_quantum/
│   ├── 01_qsvm_main_learning_curves.ipynb
│   └── 02_vqc_sigmoid_optional.ipynb
├── 04_ablations/
│   ├── 01_encoding_ablations.ipynb
│   ├── 02_training_ablations.ipynb
│   └── 03_other_sections.ipynb
├── 05_noise_hardware/
│   ├── 01_noise_study_aer.ipynb                  noise study, (c, C) re-tuned on the n = 80 training set
│   ├── 01b_noise_study_aer_frozen.ipynb          same study, (c, C) frozen at the main learning-curve settings
│   └── 02_ibm_hardware_kernel_block.ipynb        20 x 20 kernel block on ibm_fez
├── 06_analysis/
│   └── 01_final_tables_and_falsifier.ipynb
│
├── extra_sections_data/                          3x3 and 5x2.5 datasets (written by 01_data/02)
└── results/
    ├── section_constants.csv, sections_falsifier.csv, sections_ranking.csv, all_sections_learning_curves.csv
    ├── 3x2/
    │   ├── splits/            fixed_splits.json (test set and every training set), dataset_summary.json
    │   ├── classical/         learning curves of the classical baselines
    │   ├── qsvm/              QSVM and VQC learning curves, best_configs.json, sampled-kernel cost
    │   ├── ablations/         encoding, bandwidth, C, class weighting and sampling ablations
    │   ├── noise/             noise study with re-tuned (c, C)
    │   ├── noise_frozen/      noise study with frozen (c, C)
    │   ├── hardware/          final hardware block (ZZ, c = 0.02, C = 10000, job dc04d4eq4uuc73a21gd0)
    │   ├── hardware_c005/     first hardware block (ZZ, c = 0.05, C = 1000, job dbva23eps3fc73bk3eag)
    │   └── resources/         overlap-circuit resources in the simulation basis
    ├── 3x3/                   splits/, learning_curves/
    ├── 5x2_5/                 splits/, learning_curves/
    ├── figures/               every figure (200 to 300 dpi)
    └── final_tables/          report tables: table1 to table5, reproducibility.json
```

Shared code lives in `qsvm_rc/`, so a fix is made once rather than in every notebook. Each notebook starts with a cell that finds the
`Project` folder and adds it to the import path, so notebooks can be opened from their own sub-folders.

## Run order

Run each notebook top to bottom with *Restart and Run All*. Times are rough (laptop CPU unless stated).

| # | notebook | what it does | needs | time |
|---|---|---|---|---|
| 1 | `00_setup/00_environment_and_sanity_checks.ipynb` | versions, dataset files, capacity code vs the stored labels, exact NumPy kernels vs Qiskit `Statevector`, sampled vs exact kernel, noisy simulator, fake device | nothing | 1 to 2 min |
| 2 | `01_data/01_data_audit_and_fixed_splits.ipynb` | audit, fixed test set (3000 random + 1000 boundary band), every training set saved once | 1 | under 1 min |
| 3 | `01_data/02_extra_sections_generation.ipynb` | 3x3 and 5x2.5 datasets and their splits | 1 | about 1 min |
| 4 | `02_classical/01_classical_baselines.ipynb` | majority, logistic, RBF SVM, MLP, small MLP; sizes 20 to 320, seeds 0 to 4 | 2 | a few min |
| 5 | `03_quantum/01_qsvm_main_learning_curves.ipynb` | angle and ZZ QSVMs, matched-width RBF control, falsifier check, leakage check, SAFE regions; writes `best_configs.json` | 2 | a few min |
| 6 | `03_quantum/02_vqc_sigmoid_optional.ipynb` | variational classifier with sigmoid output and BCE | 2 | 5 to 15 min |
| 7 | `04_ablations/01_encoding_ablations.ipynb` | feature map, reps, entanglement, feature order, input features | 2 | a few min |
| 8 | `04_ablations/02_training_ablations.ipynb` | bandwidth sweep, C sweep, class weighting, sampling strategy | 5 | 5 to 10 min |
| 9 | `04_ablations/03_other_sections.ipynb` | full protocol on 3x3 and 5x2.5, ranking and falsifier per section | 3, 4, 5 | a few min |
| 10 | `05_noise_hardware/01_noise_study_aer.ipynb` | shots, depolarizing, readout error, fake IBM device; (c, C) re-tuned on n = 80 | 5 | about 1 h on Kaggle CPU |
| 11 | `05_noise_hardware/01b_noise_study_aer_frozen.ipynb` | same study with (c, C) from `best_configs.json` | 5 | about 1 h on Kaggle CPU |
| 12 | `05_noise_hardware/02_ibm_hardware_kernel_block.ipynb` | 20 x 20 kernel block on `ibm_fez` vs exact, noisy Aer and the fake device | 5 | see below |
| 13 | `06_analysis/01_final_tables_and_falsifier.ipynb` | final tables, figures, falsifier verdict, reproducibility info | any of the above | seconds |

Notebook 13 skips any result that is missing, and only reads saved files, so it can be re-run at any time.

## Final settings

| item | value |
|---|---|
| tuning grids | C in {0.1, 1, 10, 100, 1000, 10000}; RBF gamma in {0.001, 0.003, 0.01, 0.1, 0.25, 1, 10}; QSVM bandwidth c in {0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1} (42 settings each) |
| main angle QSVM | one Ry per qubit, c = 0.5, C = 100 (most frequent at n = 320) |
| main ZZ QSVM | `zz_feature_map(3, reps = 2, entanglement = "linear")`, c = 0.02, C = 10000 |
| hardware circuit (ibm_fez, opt level 3) | 3 qubits, depth 80, 14 CZ, 2048 shots, 210 circuits per block |

## Decisions made in the code

* **Inputs and target.** Only Pu, ex, ey are model inputs. phi Pni is never an input. It defines the boundary band and the sampling
  strategies only, and the labels always come from the dataset.
* **Scaling.** Each feature is mapped linearly from its design domain (Pu from the grid limits, ex and ey from 0 to the half dimensions)
  onto [0, pi]. Nothing is fitted on test data. The bandwidth c then multiplies the angles.
* **Fixed test set.** 3000 stratified random points plus 1000 points with |Pu / phi Pni - 1| <= 0.05. Training sets come only from the remaining pool.
* **Main metric.** Error on the boundary band (`band_error`). Also reported: F1, precision, recall, balanced accuracy and MCC on the
  random part, and the false SAFE rate (UNSAFE predicted SAFE) on both parts.
* **Matched tuning.** The RBF SVM (C x gamma) and the QSVM (C x c) each get 42 settings on the same inner stratified folds, scored on balanced accuracy.
  The grids were widened after a first run showed the ZZ bandwidth and C at the edges of the original 5 x 5 grids.
* **Noiseless kernels** are computed exactly from state vectors in NumPy (three qubits, eight amplitudes) and checked against Qiskit
  `Statevector` in notebook 00. Shots and noise enter only in the noise and hardware notebooks.
* **Noise study, two settings.** `noise/` re-tunes (c, C) on the n = 80 training set (it picks c = 0.01, a nearly flat kernel that shot
  noise alone destroys). `noise_frozen/` uses the main settings. Both are reported side by side in `table4_noise.csv`.
* **Extra sections.** 3x3 with 9 bars per face (32 bars, rho = 0.031) and 5x2.5 with 15 bars per long face and 8 per short face
  (42 bars, rho = 0.030). Layouts are set in `qsvm_rc/config.py`.
* **Capacity code.** Strain compatibility with the ACI stress block (beta1 = 0.836 for 30 MPa), displaced concrete subtracted, Bresler
  reciprocal load, phi = 0.65. It reproduces the stored main-section labels on 99.999 percent of rows.

## IBM hardware

* Credentials come from a locally saved account, `QiskitRuntimeService(name = config.IBM_ACCOUNT_NAME)`. No token is written in any notebook.
* The block is 20 points from the training pool (5 SAFE near the boundary, 5 UNSAFE near, 5 SAFE far, 5 UNSAFE far), i.e. 210 circuits
  including the diagonal, sent as one PUB in one job.
* Two blocks were run: `hardware_c005/` (c = 0.05) and `hardware/` (c = 0.02, the final ZZ setting). The bandwidth and C used are pinned in
  cell 3 of the notebook, so saved counts are always compared with the matching exact kernel.
* To reproduce without using QPU time, keep `SUBMIT_TO_HARDWARE = False` and run all: the notebook reloads `hardware_counts.json`.
* To submit a new block: back up `results/3x2/hardware/`, delete `hardware_job.json` and `hardware_counts.json`, set the settings in cell 3,
  set `SUBMIT_TO_HARDWARE = True` for one run of the submit cell, then fetch the result with the cell in section 7.

## Running on Kaggle (CPU)

1. Upload the zipped `Project` folder as a Kaggle Dataset.
2. Create a notebook from `kaggle_runner.ipynb` (everything) or `kaggle_runner_noise_frozen.ipynb` (frozen noise study and notebook 06 only),
   add the dataset as input, choose CPU and turn Internet on (for `pip install qiskit`).
3. Save Version -> Save & Run All (Commit), then download the output zip from the Output tab.

If the dataset holds more than one copy of the project, the copy cell must pick the one that contains the notebook being run.
The hardware notebook is never run on Kaggle.

## Reproducibility

Seed 0 everywhere (spaced seeds per run), training sizes 20, 40, 80, 160, 320 with seeds 0 to 4, fixed splits saved in
`results/<section>/splits/fixed_splits.json`. Library versions, grids and runtimes are written to `results/final_tables/reproducibility.json`.
