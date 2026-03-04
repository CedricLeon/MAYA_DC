# Progress tracking

## Aim EUSAR 2026

*Shared TODO list to track progress and leftover tasks.*

- [x] Get familiar with MAYA4 dataset
  - [x] See [download_maya4_data.py](scripts/download_maya4_data.py) to download samples of the dataset. The goal is to do it once and then work offline.
  - [ ] Select few specific tiles to make a dataset
    - How many tiles?
    - Spatial Splits?
- [x] Get familiar with sarpyx processing pipeline
- [x] Extract and re-implement the azimuth compression pipeline
  - [x] Ensure manual implementation generate similar results than $az$ products, see [sarpyx_azimuth_compression.py](scripts/sarpyx_azimuth_compression.py)
- [ ] Model architecture implementation
  - [x] `nn.Module` implementation
  - [ ] `LighningModule` implementation
    1. Compress/Reconstruct patches with a buffer in azimuth direction, like +500 cells each side
    2. Then perform azimuth compression
    3. Drop the buffer pixels
    4. Compute the SLC loss on the main cell area.
  - [ ] Modular loss implementation to be able to try MSE or KDE
- [ ] Dataset validation
  - [ ] Checking data quality for selected tiles
  - [ ] Ensuring focusing pipeline behavior (with an identity model)
  - [ ] Quantification of focusing error, i.e., difference between our custom azimuth focused images and the $az$ product from MAYA4.
- [ ] First experiments
  - [ ] Ensure implementation works 🙃
  - [ ] Collect reconstruction quality metrics for the dataset
  - [ ] **KDE** (Kernel Density Estimator) loss Vs. **MSE**
- [ ] Further experiments
  - [ ] Comparison of 2 architectures **Factorized Prior (FP)** against **Hyper Prior (HP)**
  - [ ] Baseline comparison?
    - Maybe I can see if I can use CompressAI to run conventional codecs to have as baselines, e.g., JPEG, JPEG2000, WebP, AV1? This will take time though.
- [ ] Results generation
  - [ ] RD-curves
  - [ ] Visualizations
- [ ] Paper writing

## Repository improvements

### Tmp TODO list

- [ ] Populate README
- [ ] Fix `MAYA4` download for some files where edge chunks get forgotten.
  - Doesn't work: "s1a-s3-raw-s-hh-20230619t153555-20230619t153611-049056-05e631.zarr" misses the last row and the last column of its chunks: `[(0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5)]`
  - Works normally: "s1a-s1-raw-s-hh-20230511t151235-20230511t151251-048487-05d521.zarr"
- [ ] When available install compressai==1.2.9 to fix opposing numpy version requirements between rasterio (>2.0) and compressai==1.2.8(<2.0). Right now numpy is 2.3.5, I manually checked it should not be a problem for compressai.
