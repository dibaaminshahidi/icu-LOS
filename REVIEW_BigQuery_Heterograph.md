# Review: `BigQuery_DB_V2.ipynb` and `Learning_Heterograph_Debug.ipynb`

## Overall verdict
The end-to-end flow is **close to workable**, but there are several **high-impact logic/scientific issues** that can bias metrics or silently corrupt graph features.

## High-severity issues

1. **Chart edge attributes are cast to integer (`torch.long`) even though they contain continuous features.**
   - In `BigQuery_DB_V2.ipynb`, `edge_attr` for `('stay','HAS_CHART','chart_concept')` is built from rates/times/stats/embeddings, but cast to `dtype=torch.long`.
   - This truncates decimals (e.g., `warning_rate`, normalized times, embeddings), destroying information and harming model learning.

2. **NeighborLoader seed handling may include non-seed nodes in train/eval loss.**
   - In `Learning_Heterograph_Debug.ipynb`, train/eval uses `seed_bs = loader.batch_size` and then slices `pred[:seed_bs]`, `target[:seed_bs]`.
   - On the final mini-batch (or any variable-sized batch), this can misalign with actual seed count; the robust approach is to use `batch['stay'].batch_size` (or `batch['stay'].n_id`/loader-provided seed count).

3. **Validation artifact path manipulation by string slicing is fragile and can save to wrong locations.**
   - `model_dir` is passed as a file path ending in `/model.pt`, then reused with `model_dir[:-8]...` and `model_dir[:-3]...`.
   - This is brittle and error-prone; use `Path(model_dir).with_name(...)` or explicit directory joins.

4. **`log_mape` can divide by zero after `log1p` when target is zero-length-of-stay in transformed scale.**
   - `log_mape` divides by `np.log1p(y_true)` with no epsilon guard.
   - If `y_true == 0`, denominator is 0 and metric becomes inf/NaN.

## Medium-severity issues

1. **Likely filename typo when reloading selected score parquet.**
   - Writes `icu_stay_selected_score.parquet` but later reads `icu_stay_selected_score_.parquet`.
   - This can break pipeline reproducibility or accidentally load stale files.

2. **Potential index misalignment when recomputing `value` column for text chart events.**
   - After filtering/subsetting `text_chartevents`, code assigns:
     `text_chartevents.loc[:,"value"] = text_chartevents_encoded['value_clean'].apply(_extract_number)`
   - This depends on index alignment between different DataFrames and can silently produce NaNs/wrong mapping.

3. **Inconsistent random seed between loader creation calls in one run block.**
   - First call uses `seed=12`, later call uses `seed=42` in the same experiment block.
   - This weakens experiment traceability and reproducibility.

4. **Misspelled output file name `peredicts.pkl`.**
   - Not scientifically wrong, but increases risk of downstream loading mistakes.

## Recommended fixes (priority order)

1. Use `edge_attr` with `dtype=torch.float32` for chart edges.
2. Replace `seed_bs = loader.batch_size` with per-batch seed count from the batch object.
3. Replace all `model_dir[:-k]` path slicing with `pathlib.Path` operations.
4. Add epsilon to `log_mape` denominator, e.g., `denom = np.maximum(np.log1p(y_true), 1e-8)`.
5. Standardize parquet filenames and run a quick existence check before load.
6. Ensure `value` extraction uses the same filtered DataFrame source to avoid index drift.
7. Use one fixed seed per experiment block and log it in saved metadata.

## Scientific interpretation impact

- The `torch.long` cast on continuous edge attributes is the most damaging bug: it can substantially reduce model signal and mislead conclusions about heterograph utility.
- Seed slicing assumptions in mini-batch training/eval can bias validation/test metrics.
- Metric instability (log-MAPE division) can make reported evaluation unreliable for shorter stays.

