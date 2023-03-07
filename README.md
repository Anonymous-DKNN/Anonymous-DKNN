# Denoised k-nearest neighbor (DKNN)

Anonymous Submission #4835 to ICCV 2023.

# Installation

#### Step 1. Install libraries

- Install libraries in [requirements.txt](requirements.txt)
- If the code malfunctions, try this: [requirements_all.txt](requirements_all.txt).   

#### Step 2. Download files

- [Download features](https://drive.google.com/file/d/14Gqj2X1OmL64hSgTUMLnCDxdZCk7-mue/view?usp=share_link)
- [Download meta data](https://drive.google.com/file/d/1IdVelWK9cGuaaLczotPxCx1V2elAdWBd/view?usp=share_link)
- [Download all 6 files of patches in this folder](https://drive.google.com/drive/folders/1iNGFY7fuOG4bCrbQAF1-VE9aI7BdC1zi?usp=sharing)

#### Step 3. Untar all *.tar.xz files and place properly

```
Anonymous-DKNN
-- patches
-- features
``` 

#### Step 4. Generate pseudo anomaly scores

```
./run1_pseudo_anomaly_scores.sh
```

#### Step 5. Evaluate on each dataset

```
./run2_evaluate.sh ped2
./run2_evaluate.sh avenue
./run2_evaluate.sh shanghaitech
```
