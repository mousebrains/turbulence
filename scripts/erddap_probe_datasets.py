import json
import urllib.error
import urllib.request

BASE = "https://slocum-data.marine.rutgers.edu/erddap"


def search(term):
    url = f"{BASE}/search/index.json?searchFor={term}&page=1&itemsPerPage=100"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return f"HTTPError {e.code} {e.headers.get_content_type()}"
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:100]}"
    tbl = d["table"]
    i = tbl["columnNames"].index("Dataset ID")
    return sorted({row[i] for row in tbl["rows"]})


for term in ("ru33-20211001T1841", "ru33"):
    res = search(term)
    print(f"--- searchFor={term} ---")
    if isinstance(res, str):
        print("   ", res)
    else:
        print(f"    {len(res)} dataset(s)")
        for x in res[:25]:
            print("     ", x)
