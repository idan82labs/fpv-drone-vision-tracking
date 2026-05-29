# Local Models

`models/` is reserved for promoted runtime model files and calibration assets.
The files in this directory are ignored by Git by default.

Promotion rule:

1. Train and audit models under `artifacts/`.
2. Promote a model only after leave-one-clip-out or held-out evaluation is
   documented in `docs/STATUS.md`.
3. Record the model filename, source artifact, checksum, and intended runtime
   profile in a small tracked document or release note.
4. Include the model in a deployment bundle only through an explicit bundle
   rule, not by copying all local artifacts.

The current Raspberry Pi live path avoids sklearn/learned-ranker inference in
the hot loop. Do not add a model to the Pi path until runtime and behavior are
measured.
