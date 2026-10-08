# Baselines under the same protocol

The paper reproduces VM-UNet and VMKLA-UNet with their official code. To make the comparison
controlled, use the split files and preprocessed data of this repository, the same three seeds
(42, 123, 256), and score all predictions with this repository's evaluation code.

1. **Data and splits.** Train each baseline on the volumes listed in `splits/<dataset>_train.txt`,
   select checkpoints on `splits/<dataset>_val.txt`, and never use the test split for selection.
   The preprocessed slices in `data/<dataset>/slices/` and volumes in `data/<dataset>/volumes/`
   can be read by any PyTorch pipeline (`adapmamba/data.py` shows the format).
2. **Training settings.** Keep each baseline's published optimiser and schedule unless the paper
   states otherwise, and record the settings used.
3. **Predictions.** For every test volume, save the predicted label volume on the ground-truth grid
   as `<pred_dir>/<volume_id>.npz` with key `pred` (uint8, shape D x H x W). For the significance
   tests, average the softmax outputs of the three seeds before taking the argmax, as for
   AdapMamba-UNet (`adapmamba/metrics.py: predict_volume_probs`).
4. **Scoring.**
   ```bash
   python test.py --dataset synapse --data_root data/synapse \
       --pred_dir preds/vmkla_ensemble --output_dir results/synapse/vmkla_unet
   python stats.py --a results/synapse/full/ensemble_cases.csv \
       --b results/synapse/vmkla_unet/external_cases.csv \
       --name_a AdapMamba-UNet --name_b VMKLA-UNet --out results/synapse/stats_vs_vmkla
   ```
   For per-seed baseline results, score each seed's predictions separately in the same way.

Official code: VM-UNet, https://github.com/JCruan519/VM-UNet ; VMKLA-UNet, see the code
availability statement of its paper (Scientific Reports 15, 13258, 2025).
