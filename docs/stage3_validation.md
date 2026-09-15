# Stage 3 validation

Stage 3 exact import and alignment passed on CPU. The 40 pinned trajectories produced
6,474 released chunks with zero unresolved spans. Outputs are checksummed Parquet under
`/lambda/nfs/preanchor/preanchor/data/processed/stage3/`:

- `trajectories.parquet`: 40 rows;
- `sentences.parquet`: 6,474 rows;
- every stored span slices exactly the corresponding `full_cot` text;
- repeated chunks align sequentially, and whitespace-tolerant matches are marked.

The 4682 smoke import, schema round trips, exact-span and repeated-text tests passed.
Full suite after this stage: 66 tests, with the two existing Typer/Click warnings.
No model inference, test metric or GPU workload was run.
