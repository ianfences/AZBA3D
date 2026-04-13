import os
import json
import re
import numpy as np
import nibabel as nib
from skimage import measure
from skimage.morphology import closing, ball
import trimesh

DATA_DIR          = r"C:\Users\Ianbe\zfish"
OUTPUT_DIR        = r"C:\Users\Ianbe\zfish\web_output"
SEG_FILE          = "2021-08-22_AZBA_segmentation.nii.gz"
LABEL_FILE        = "2021-08-22_AZBA_Label_descriptions.txt"
MIN_VOXELS        = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Reading label descriptions...")
labels = {}
with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+", line, maxsplit=7)
        if len(parts) < 8:
            continue
        try:
            idx = int(parts[0])
            r, g, b = int(parts[1]), int(parts[2]), int(parts[3])
            name = parts[7].strip().strip('"')
            labels[idx] = {"name": name, "r": r, "g": g, "b": b}
        except ValueError:
            continue
print(f"  Loaded {len(labels)} region labels")

print("\nLoading segmentation volume...")
seg_img = nib.load(os.path.join(DATA_DIR, SEG_FILE))
seg_data = np.round(seg_img.get_fdata()).astype(np.int32)
voxel_size = seg_img.header.get_zooms()
print(f"  Shape: {seg_data.shape}, Voxel size: {voxel_size}")

region_ids = sorted([i for i in np.unique(seg_data) if i > 0])
print(f"  Found {len(region_ids)} non-background regions\n")

# Compute global center ONCE so all regions share the same coordinate space
global_center = (np.array(seg_data.shape) / 2.0) * np.array(voxel_size)

manifest = []
failed = []

for i, rid in enumerate(region_ids):
    label_info = labels.get(rid, {"name": f"Region_{rid}", "r": 180, "g": 180, "b": 180})
    name = label_info["name"]
    safe_name = re.sub(r"[^\w\-]", "_", name)
    fname = f"region_{rid:04d}_{safe_name}.glb"
    out_path = os.path.join(OUTPUT_DIR, fname)

    # Skip if already converted
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        size_kb = os.path.getsize(out_path) / 1000
        r2, g2, b2 = label_info["r"], label_info["g"], label_info["b"]
        mask = (seg_data == rid)
        n_vox = int(mask.sum())
        manifest.append({"id": rid, "name": name, "file": fname, "color": [r2, g2, b2], "voxels": n_vox, "faces": 0})
        print(f"  [{i+1}/{len(region_ids)}] {name} — already done ({size_kb:.1f} KB), skipping")
        continue

    print(f"  [{i+1}/{len(region_ids)}] {name} (id={rid})", end="", flush=True)

    mask = (seg_data == rid)
    n_vox = int(mask.sum())

    if n_vox < MIN_VOXELS:
        print(f" — skipped ({n_vox} voxels, too small)")
        continue

    mask = closing(mask, ball(1))

    try:
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=2)
        verts = verts * np.array(voxel_size)
        verts -= global_center

        # Fix NIfTI axis orientation for Three.js
        # NIfTI shape is (470, 1224, 670) = (Z, Y, X) in anatomical space
        # Remap so Three.js X=left-right, Y=dorsal-ventral, Z=anterior-posterior
        verts = verts[:, [1, 2, 0]]   # Y becomes long axis, Z becomes dorsal
        verts[:, 2] *= -1              # flip to correct orientation

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        # No decimation — export as-is (step_size=2 already reduces detail)

        if len(mesh.faces) == 0:
            print(f" — skipped (empty mesh after decimation)")
            continue

        r, g, b = label_info["r"], label_info["g"], label_info["b"]
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=np.array([[r, g, b, 220]] * len(mesh.vertices), dtype=np.uint8)
        )

        safe_name = re.sub(r"[^\w\-]", "_", name)
        fname = f"region_{rid:04d}_{safe_name}.glb"
        out_path = os.path.join(OUTPUT_DIR, fname)
        mesh.export(out_path)
        size_kb = os.path.getsize(out_path) / 1000

        manifest.append({
            "id": rid,
            "name": name,
            "file": fname,
            "color": [r, g, b],
            "voxels": n_vox,
            "faces": len(mesh.faces),
        })
        print(f" — {len(mesh.faces)} faces, {size_kb:.1f} KB")

    except Exception as e:
        print(f" — ERROR: {e}")
        failed.append({"id": rid, "name": name, "error": str(e)})

manifest_path = os.path.join(OUTPUT_DIR, "regions.json")
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\n{'='*60}")
print(f"Done! {len(manifest)} meshes exported to: {OUTPUT_DIR}")
if failed:
    print(f"Failed ({len(failed)}): {[f['name'] for f in failed]}")
total_mb = sum(os.path.getsize(os.path.join(OUTPUT_DIR, m["file"])) for m in manifest) / 1e6
print(f"Total output size: {total_mb:.1f} MB")
print(f"{'='*60}")
