# Elite: GPU-Accelerated Dynamic Object Removal for LiDAR SLAM
<img width="2192" height="1298" alt="image" src="https://github.com/user-attachments/assets/00095838-95b2-4eb8-86ca-99e479195e27" />

This repository contains an optimized implementation of **Elite**, a LiDAR-based
dynamic object removal pipeline designed for **SLAM and static map construction**.
The original pipeline targets large-scale autonomous driving datasets
(e.g., SemanticKITTI) but suffers from severe runtime bottlenecks due to CPU-bound
neighbor search and loop-based updates.

This work focuses on **system-level acceleration** while preserving mapping quality.

---

## Motivation

Dynamic objects (cars, pedestrians) degrade SLAM map quality if not properly filtered.
Elite addresses this by estimating per-point dynamic likelihoods across LiDAR scans.

However, the original implementation:
- Relies on **CPU-based KDTree kNN**
- Uses **nested Python loops** for Bayesian updates
- Scales poorly to million-point LiDAR scans

This makes large-scale deployment impractical.

---

## Key Contributions

- **GPU acceleration** using PyTorch + CUDA
- **Vectorized Bayesian updates** in logit space (eliminating per-point loops)
- **Approximate kNN search** via **FAISS HNSW**
- **Voxel-hash based downsampling** to reduce redundant LiDAR points
- Optimized CPU–GPU data flow to remove synchronization bottlenecks

The pipeline is refactored to operate efficiently on large point clouds
while maintaining the original probabilistic formulation.

---

## Performance

### Runtime
- **End-to-end speed-up**: ~**17×** (≈28–30s → ≈1.7s per scan)

> In practice, peak speed-ups exceeding **~18–20×** are observed depending on
> caching, voxel resolution, and kNN parameters.
## Accuracy and Evaluation

Acceleration is achieved **without degrading dynamic object removal quality**.
We evaluate the optimized pipeline using the official **SemanticKITTI dynamic
object removal benchmark**, following prior work (e.g., SuMa, Removert, ERASOR).

### Metrics
We report standard evaluation metrics:
- **SA (%)**: Static Accuracy  
- **DA (%)**: Dynamic Accuracy  
- **AA (%)**: Average Accuracy  
- **HA (%)**: Harmonic Accuracy  

## Benchmark Results

We evaluate the optimized ELite implementation using the **DynamicMap benchmark**
on **SemanticKITTI**[1, 2], reporting map-level static/dynamic separation accuracy.

### Sequence 00

| Metric | Value |
|------|------|
| # Static Points | 15,659,894 |
| # Dynamic Points | 1,702,336 |
| Static Accuracy (SA) | 90.65 % |
| Dynamic Accuracy (DA) | 92.01 % |
| Average Accuracy (AA) | 91.33 % |
| Harmonic Accuracy (HA) | 91.33 % |
| Runtime | **1.08 s / scan** |

The results are obtained from the final cleaned static map produced after
dynamic object removal.


### Summary

Despite a ~17×–20× runtime reduction, the optimized implementation:
- Preserves **dynamic/static classification accuracy**
- Maintains **stable F1 / harmonic accuracy**
- Produces SLAM-ready static maps without additional artifacts

This confirms that system-level optimization does not compromise mapping quality.

## References

[1] H. Gil, D. Lee, G. Kim, and A. Kim,  
    *Ephemerality Meets LiDAR-based Lifelong Mapping*,  
    arXiv:2502.13452, 2025.

[2] Q. Zhang et al.,  
    *A Dynamic Points Removal Benchmark in Point Cloud Maps*,  
    IEEE ITSC, 2023.


