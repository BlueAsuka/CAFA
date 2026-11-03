OptMATH-Train -> NL-Gurobi Contrastive Dataset Builder

1) Python 3.10+
2) Install: python -m pip install -U datasets huggingface_hub tqdm
3) Test offline: python optmath_nl_gurobi_builder.py --demo --out demo_cl
4) Small subset via streaming: python optmath_nl_gurobi_builder.py --limit 2000 --negatives 5 --augment 2 --out clip_small
5) Full dataset: python optmath_nl_gurobi_builder.py --mode download --limit 0 --negatives 5 --augment 2 --out clip_full

Output: pairs_{train,val,test}.jsonl, ranking_{train,val,test}.jsonl, stats.json.

Progress bars are enabled by default:
- `--mode download` shows downloaded bytes, transfer rate, and ETA when a download is needed, followed by a curation bar. Cached files are reused, so an already-downloaded file may not display a transfer bar.
- Curation shows inspected source records, processing rate, and live counts for accepted records, skipped records, and generated negatives. A known record total or `--limit` also provides percentage and ETA.
- In the default streaming mode, downloading and curation happen together; the record progress bar tracks their combined progress. Full streams/local JSONL files with no known record total show counts and rate without an estimated percentage.
- Add `--no-progress` to suppress the builder's progress bars. The final statistics and output files remain available.

Source repo: https://huggingface.co/datasets/Aurora-Gem/OptMATH-Train
Uses source file OptMATH-Train.jsonl, keys instruction/input/output/answer.

WARNING: Never executes downloaded Gurobi code. Its positive pairs reflect source reference status only, not independently verified semantic fidelity. AST-generated negative examples are NOT guaranteed logically different; validate before treating them as gold negatives. The source dataset card currently reports license unknown; review usage rights.
