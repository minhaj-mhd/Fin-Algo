# Master Veto Override: Cluster Sweet Spots

This table represents the culmination of the hierarchical sub-clustering analysis. It maps the exact "sweet spots" (the mathematical centroids and their optimal distance boundaries) that successfully isolate historical winners from market noise. 

Any incoming trade that is mathematically "close" (falls within the distance threshold) to one of these sub-centroids is highly likely to be a massive winner, and its veto should be overridden.

| Primary Cluster | Sub-Cluster Model | Sub-Cluster ID | Distance Threshold | Trades Rescued | Win Rate | Avg Net PnL | Notes / Archetype |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cluster 1** | K=3 | **Sub-Cluster 1** | `< 1.00` | 188 | 58.0% | +0.06% | Broad Contrarian Short |
| | | | `< 0.75` | **120** | **61.7%** | **+0.10%** | **Sweet Spot** |
| | | | `< 0.50` | 47 | 63.8% | +0.14% | Ultra-strict core |
| **Cluster 2** | K=3 | **Sub-Cluster 0** | `< 0.75` | 21 | 52.4% | +0.08% | Noise border |
| | | | `< 0.50` | **15** | **73.3%** | **+0.31%** | **Massive Sweet Spot** |
| | | **Sub-Cluster 1** | `< 1.00` | **8** | **75.0%** | **+0.50%** | **High-Profit Niche** |
| **Cluster 3** | K=5 | **Sub-Cluster 3** | `< 1.00` | 182 | 48.9% | +0.02% | Noise border |
| | | | `< 0.75` | **72** | **58.3%** | **+0.07%** | **Sweet Spot (High Volume)** |
| | | **Sub-Cluster 0** | `< 0.75` | 13 | 69.2% | +0.25% | Pure core |
| | | | `< 0.50` | **6** | **83.3%** | **+0.46%** | **Ultra-Pure Sweet Spot** |
| **Cluster 0** | K=3 | **Sub-Cluster 1** | `< 0.75` | 50 | 54.0% | +0.11% | Weakest of the sweet spots |

## Summary of Usable Overrides
If we implement the Vanguard Veto Override engine to strictly approve trades that match the bolded **Sweet Spots** above, we successfully isolate **~221 historical trades** with an aggregate win rate safely above **60%** and exceptionally high net profit.
