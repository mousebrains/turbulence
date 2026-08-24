import re
import urllib.error
import urllib.request

BASE = "https://slocum-data.marine.rutgers.edu/erddap/tabledap"


def variables(ds):
    try:
        with urllib.request.urlopen(f"{BASE}/{ds}.dds", timeout=30) as r:
            txt = r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return f"HTTPError {e.code} ({e.headers.get_content_type()})"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:80]}"
    return re.findall(r"^\s+\w+\s+(\w+);", txt, re.M)


for ds in ("ru33-20211001T1841-trajectory-raw-delayed",
           "ru33-20211001T1841-profile-sci-delayed"):
    v = variables(ds)
    print(f"--- {ds} ---")
    if isinstance(v, str):
        print("   ", v)
        continue
    print(f"    {len(v)} variables")
    interesting = [x for x in v if any(k in x for k in
                   ("temp", "cond", "press", "time", "qartod", "qc", "flag", "lat", "lon"))]
    for x in interesting[:22]:
        print("     ", x)
