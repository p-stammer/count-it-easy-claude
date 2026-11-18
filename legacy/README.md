# Legacy: Kivy app (2022)

The original Android app (Kivy/KivyMD + camera4kivy + TFLite) and its Colab
training notebook, kept for reference. It is **not** maintained; the current
app lives in [`/web`](../web), the ML pipeline in [`/ml`](../ml).

- `kivy-app/` – `main.py`, Buildozer specs, CameraX provider, overlay mask
- `models/` – the three original 3-class models (`.h5`, `.tflite`);
  ONNX conversions are in `ml/models/legacy/` for benchmarking
- `training/Count_It_Easy.ipynb` – original training notebook (Keras);
  its board labels were migrated to `data/boards.json` (three labelling
  errors fixed, see notes there)

Known issues that motivated the rewrite (all fixed in v2):

1. Crash on *Recount* after a profile's first game (`np.max([])`).
2. Adding a profile re-appended the last score to the previous profile.
3. Recount in single mode during an active Best-of-3 corrupted that session.
4. Recount did not roll back a wrongly counted highscore.
5. Names containing commas corrupted `highscores.txt`.
6. Pressing the shutter before the first camera frame crashed.
7. User data was stored inside the app directory (lost on updates).
8. KivyMD installed from `master` (unpinned) – current KivyMD 2.x no longer builds.
9. `android.api = 32` – Google Play now requires a much newer target API.
10. No iOS path (no `tflite-runtime` for iOS).
