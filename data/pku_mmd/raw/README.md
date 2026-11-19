# PKU-MMD raw data (not committed to git)

Download from the official project page (academic request required):
http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html

Place the extracted files here so this folder looks like:

```
data/pku_mmd/raw/
    Skeleton/
        0002-L.skeleton
        0002-M.skeleton
        ...
    Label/
        0002-L.txt
        0002-M.txt
        ...
```

- `Skeleton/*.skeleton` — one line per frame, 3 x 25 x 2 floats (25 Kinect v2 joints, 3D xyz, up to 2 tracked bodies).
- `Label/*.txt` — one row per action instance: `label_id, start_frame, end_frame, confidence`.

This folder is in `.gitignore` — the raw dataset is too large (and access-restricted) to commit. Only the code that reads it lives in version control.
