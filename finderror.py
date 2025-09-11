# show missing files (Linux one-liner)

import pandas as pd, os
df=pd.read_csv("labels.csv")
missing=[]
for p in df['image_path']:
    rel=str(p).lstrip("./")
    probe=rel if rel.startswith("dataset/") else os.path.join("dataset", rel)
    if not os.path.exists(probe): missing.append(probe)
print(f"Missing {len(missing)} files:")
for m in missing: print(" -", m)

