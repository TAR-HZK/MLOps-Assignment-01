# simple uniqueness check: expects data/ contains group_x/ with dataset_id.txt inside
import glob, sys

ids = {}
for f in glob.glob("data/*/dataset_id.txt"):
    with open(f) as fh:
        idv = fh.read().strip()
    if idv in ids:
        print("Duplicate dataset id", idv, "in", ids[idv], "and", f)
        sys.exit(1)
    ids[idv] = f

print("No duplicate dataset ids found.")
