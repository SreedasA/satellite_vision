# Phase 2 experiment configs

Each config changes **one thing** relative to the one before it (ROADMAP: one change at a time).
All inherit from `../dav2s_base.yaml` via `base:`.

| Config | Change vs previous | Question it answers |
|---|---|---|
| `a_long.yaml` | 30 epochs, cosine LR with 1-epoch warm-up, tall-aware checkpoint selection | Does longer training fix tall-building bias? |
| `b_tall_weight.yaml` | + height-weighted L1 (`w_tall: 1.0`) | Does weighting tall pixels reduce the bias? |
| `c_higher_lr.yaml` | + 2x encoder and head LR | Is the model under-trained at the base LR? |
| `d_mse.yaml` | + MSE term (`w_mse: 0.05`) | Does penalising large errors help tall buildings? |
| `unet_baseline.yaml` | U-Net (ResNet-34, ImageNet) instead of DAv2, same 30-epoch schedule | Pipeline sanity check vs published U-Net (~2.98 m RMSE on DFC19) |

Run (DFC23 block is the primary benchmark; repeat on DFC19 tile for comparison with published work):

    scripts\train.cmd --config configs/train/p2/a_long.yaml --dataset dfc23 --split block --run-name p2a_dfc23_block
