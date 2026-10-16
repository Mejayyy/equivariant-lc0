# Equivariance visualisation app

A Streamlit app for **seeing** equivariance on a chess board. For a position
x and a group element g it shows, for every model, the output on the
transformed board, **f(g·x)**, next to the transformed output, **g·f(x)**.
Equivariant models make the two coincide; the others visibly don't:

| Model | σ_h (file mirror) | σ_c (colour swap + rank flip) | σ_h∘σ_c (180°) |
|---|:-:|:-:|:-:|
| `gcnn` (G_chess) | MATCH | MATCH | MATCH |
| `sgcnn` ({e, σ_c}) | – | MATCH | – |
| `fgcnn` ({e, σ_h}) | MATCH | – | – |
| `fgcnn2` ({e, σ_h}, AZ-layout policy head) | MATCH | – | – |
| `baseline` | – | – | – |

The baseline's residual shows how much equivariance it picked up from the
canonicalised training data. The badges use the same per-head relative error
as `metrics/equivariance.py`, so the picture and the numbers agree.

## Run

From the repository root, in the project's environment (or after
`pip install -r app/requirements.txt`):

```bash
streamlit run app/app.py
```

The initial state can be set from the URL, e.g. `?fen=<URL-encoded FEN>&g=sigma_h`
(or `sigma_c`, `sigma_h_c`), so a position can be shared as a link.

## Modes

- **Equivariance.** Three boards show your position, the network's input x
  and the transformed input g·x. Below, each model draws its top-5 moves on
  g·x, blue for the actual f(g·x) and red for the expected g·f(x), with a
  MATCH / MISMATCH badge, per-head residuals, both WDL bars and moves-left.
  A panel at the bottom runs `metrics/equivariance.py` over random inputs.
- **Prediction.** Each model's top-5 legal moves, win/draw/loss, eval
  Q = P(win) − P(loss) and moves-left for the current position. Moves can be
  played from the sidebar.

## Design choices

- **Boards are drawn in the canonical frame** (the side to move's pieces as
  White at the bottom), the network's actual input space. Converting back to
  the real orientation would undo σ_c's colour swap and rank flip and make
  g·x look identical to x.
- **g·x comes from `symmetry.actions.group_action_input`**, the action the
  models are equivariant to, rather than a hand-made board flip. A
  transformed position may be illegal as chess; it is there to show the
  transform.
- **Arrows are restricted to legal moves.** The legal moves of g·x are g
  applied to the legal moves of x, since g permutes the 1858-move space.
  The badge and residuals still use the full policy vector.
- **Moves follow LCZero's policy encoding**, as in the training targets:
  castling is "king takes own rook" (e1h1) and a knight promotion is the
  plain pawn move (e7e8).
- **History planes.** A pasted FEN has no move history, so the 1-ply-ago
  planes are filled with the current position (or zeros, from the sidebar)
  and predictions at a root FEN are approximate. Played moves fill in the
  real history.
- **MATCH threshold 1e-4**: far above the ~1e-7 of the equivariant models
  and far below the 0.1–1 of a genuine mismatch.

## Checkpoints

`checkpoints/` holds small versions of the four thesis models, trained for
20k steps, and the fgcnn2 checkpoint from the run reported in the main
README (42k steps). Only the weights are kept, stored in float16 (they are
cast back to float32 when loaded):

| File | Model | Blocks × filters | Params |
|---|---|---|---:|
| `baseline.pt` | baseline | 5 × 128 | 2.20 M |
| `gcnn.pt` | gcnn | 5 × 43 | 2.07 M |
| `sgcnn.pt` | sgcnn | 5 × 90 | 2.44 M |
| `fgcnn.pt` | fgcnn | 5 × 90 | 2.44 M |
| `fgcnn2.pt` | fgcnn2 | 10 × 88 | 3.85 M |

Each model is rebuilt from its checkpoint's own config. The checkpoint
folder in the sidebar (relative to the repository root) also accepts full
training checkpoints, e.g. `runs/<run>`; the latest checkpoint per model is
used.

## Checks

```bash
python app/verify_equivariance.py   # metric + on-board verdicts for the shipped checkpoints
pytest tests/test_app.py            # encoder round trip, move indexing, checkpoint symmetries
```

## Files

```
app.py                  Streamlit UI (equivariance and prediction modes)
verify_equivariance.py  headless version of the equivariance checks
viz/encoding.py         FEN -> 34 input planes; canonical-frame rendering
viz/modelio.py          model loading, prediction, f(g·x) vs g·f(x) (no Streamlit)
viz/render.py           board SVGs, WDL bars, badges
checkpoints/            the five models
```
