Short guidance for AI contributors working on FastDetect

* Big picture
  * This repository is a set of offline signal-processing tools to detect and decode LoRa-like chirped symbols from complex64 binary captures.
  * Key phases: pre-detection sweep (in `pre_detect.py` / `work.py` / `work_new.py`) -> symbol timing and CFO refinement (`work_new.py`, `symbtime.py`) -> payload decode (`aft_decode.py`).
  * Data flows: binary complex64 files (raw I/Q) are read by `reader.SlidingComplex64Reader.get(start, length)` and then processed by dechirping/FFT helpers in `pre_detect.py`/`utils.py` and higher-level `work*` modules.

* Important files to read first
  * `Config.py` — central parameters (SF, BW, fs, preamble length, decode matrices). Many parts derive behavior from these constants.
  * `utils.py` — backend abstraction (NumPy/CuPy), logging, helper math (to_host/to_device/around). Respect `USE_GPU` variable.
  * `pre_detect.py` — detection sweep, FFT-based up/down chirp pairing, and the generator `updown_gen` used by `work.py`/`work_new.py`.
  * `work_new.py` — higher-level fitting/refinement pipeline (fitcoef*, symbtime usage) — good example for time/frequency refinement changes.
  * `aft_decode.py` — decoding matrices and `decode_payload` (how symbols are mapped back to codes).

* Run / debug
  * The project is executed from `main.py` which calls `work_new(...)` or `work(...)`. Run with Python 3.10+.
  * To enable GPU path set environment variable: `export USE_GPU=1` (requires CuPy). Default uses NumPy/Scipy.
  * Example local run (CPU):
    * export USE_GPU=0
    * python3 main.py
  * Logging writes to `run_250828.log` via `utils.logger`.

* Coding patterns & conventions
  * Backend-agnostic arrays: use `xp` from `utils.py` (alias for numpy or cupy). Convert to host arrays with `to_host()` and scalars with `to_scalar()`.
  * File I/O: read raw IQ as complex64 elements via `SlidingComplex64Reader.get(start, length)` — arguments are in complex-element units (not bytes).
  * FFT helpers: prefer `myfft(...)` / `dechirp_fft(...)` in `pre_detect.py` to ensure consistent plan/fftshift semantics.
  * Use `Config` constants rather than hard-coded numbers; many algorithms assume `Config.nsamp`, `Config.fft_n`, and `Config.sf`.

* Tests & quick checks
  * No automated tests found; validate changes by running `python3 main.py` on a small data file under `data/` and observing printed selected candidates and plots.
  * For numerical changes, add small reproducible harnesses similar to `work_new.fitcoef2` that call `reader.get(...)` on short windows.

* Common pitfalls for code edits
  * Mixing NumPy/CuPy: always use `xp`/`to_device`/`to_host` helpers — avoid direct `numpy` imports unless in `utils.py` fallback.
  * Time/frequency units: many functions use samples vs seconds vs symbol indices. Check `Config.tsig`, `nsamp` and `tsymblen` math before changing formulas.
  * Reader offsets: `reader.get(start, length)` expects start in complex-sample units; convert from seconds by multiplying with sample rate (`Config.fs`) or using existing helpers in `work_new.py`.

* Example edits an agent might perform
  * Add an optional buffered reader: extend `SlidingComplex64Reader` with a seek/cache layer and reuse in `work_new.py` and `pre_detect.py`.
  * Add unit-conversion helpers: small functions to convert (samples <-> seconds <-> symbol indices) and replace repeated expressions in `work_new.py`/`pre_detect.py`.
  * Move common FFT/plan initialization into `Config` (already partly present) — maintain `Config.plan` usage.

* Where to ask when unclear
  * If unsure about expected detectors thresholds or plotting behavior, inspect `work.py`/`work_new.py` usage traces and logs (`run_250828.log`).

If anything here is unclear or you'd like more examples (e.g., a suggested unit-test harness), tell me which area to expand.
