import os, pickle, re, time
import pandas as pd
import xml.etree.ElementTree as ET
 
EXCLUDE_SECTIONS = {"Tools_Configuration", "LM_Configurations", "Generation_Parameters", "LM_Memorisations"}
NULL_TOKENS = {"", "none", "null", "nan"}
 
 
def _clean(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = v.strip()
        return v if v != "" else None
    return v
 
 
NOT_DEFINED = "_Not_Defined"      # placeholder tags such as PSD_ID_Not_Defined  (= "no reference")
 
 
def _flatten(elem, prefix, out):
    """Flatten the children of one XML object into {column: [values]}.
       <Track_ID>2</Track_ID>                    -> Track_ID            = 2
       <Kp Value="5" Corrected_Gap_Value="0"/>   -> Kp.Value = 5, Kp.Corrected_Gap_Value = 0
       <ESP_ID_List><ESP_ID>2</ESP_ID>..         -> ESP_ID_List         = "2;7"   (list)
       <X_List><X Value="1" ID="3"/>..           -> X_List/X.Value = "1;..", X_List/X.ID = ".."
       <PSD_ID_Not_Defined>0</..>                -> PSD_ID              = empty   (no reference)"""
    for k in elem:
        tag, kids = k.tag, list(k)
        if not kids:
            if k.attrib:
                for a, v in k.attrib.items():
                    out.setdefault(f"{prefix}{tag}.{a}", []).append(v)
            elif tag.endswith(NOT_DEFINED):
                out.setdefault(prefix + tag[: -len(NOT_DEFINED)], [])
            else:
                out.setdefault(prefix + tag, []).append((k.text or "").strip())
        elif all((not g.attrib and not list(g)) for g in kids):          # simple list
            out[prefix + tag] = [(g.text or "").strip() for g in kids if not g.tag.endswith(NOT_DEFINED)]
        else:                                                           # structured list / group
            _flatten(k, prefix + tag + "/", out)
 
 
def load_xml(path):
    """SyDB XML -> {section name: DataFrame}. Section = first-level element (e.g. Signals);
    one row per object inside it. Attributes ID and Name become columns '.ID' and '.Name'."""
    root = ET.parse(path).getroot()
    raw = {}
    for sec in root:
        objs = list(sec)
        if not objs:
            continue
        rows = []
        for o in objs:
            d = {}
            for a, v in o.attrib.items():
                d["." + a if a in ("ID", "Name") else a] = [v]
            _flatten(o, "", d)
            rows.append({k: (";".join(v) if v else None) for k, v in d.items()})
        raw[sec.tag] = pd.DataFrame(rows)
    return raw
 
 
def apply_hygiene(raw):
    tables = {}
    for name, df in raw.items():
        if name in EXCLUDE_SECTIONS or len(df) == 0:
            continue
        seen, names = {}, []
        for c in df.columns:
            c = str(c)
            if c in seen:
                seen[c] += 1
                c = f"{c}__{seen[c]}"
            else:
                seen[c] = 0
            names.append(c)
        d = df.copy()
        d.columns = names
        d = d.astype(object).apply(lambda s: s.map(_clean))
        d = d.apply(lambda s: s.map(lambda v: None if (isinstance(v, str) and v.lower() == "none") or
                                    (isinstance(v, float) and v != v) else v))
        d = d[d.notna().any(axis=1)].reset_index(drop=True)
        tables[name] = d.astype(object).where(d.notna(), None)
    return tables
 
 
def load_sydb(path, cache=True, verbose=True):
    """Load one SyDB XML -> clean tables. A pickle cache (<file>.sydbcache) makes re-runs fast."""
    cache_file = path + ".sydbcache"
    stamp = (os.path.getsize(path), int(os.path.getmtime(path)))
    if cache and os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                blob = pickle.load(f)
            if blob.get("stamp") == stamp:
                return blob["tables"]
        except Exception:
            pass
    t0 = time.time()
    tables = apply_hygiene(load_xml(path))
    if verbose:
        print(f"[loader] {len(tables)} sections, {sum(len(d) for d in tables.values())} rows "
              f"read in {time.time()-t0:.1f}s")
    if cache:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"stamp": stamp, "tables": tables}, f)
        except Exception:
            pass
    return tables
 
 
# ---------- small value helpers used everywhere ----------
def is_null(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return isinstance(v, str) and v.strip().lower() in NULL_TOKENS
 
 
def is_ref_null(v):
    """null or the SyDT placeholder '0' / 'None' used for 'no link'."""
    if is_null(v):
        return True
    s = str(v).strip()
    return s in ("0", "0.0", "None", "none")
 
 
def to_num(v, default=None):
    if is_null(v):
        return default
    if isinstance(v, bool):
        return float(v)
    try:
        return float(str(v).replace(",", ".").split("::")[0])
    except Exception:
        return default
 
 
def to_bool(v):
    """True / False / None (unknown)."""
    if isinstance(v, bool):
        return v
    if is_null(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None
 
 
def split_list(v):
    """'A;B;C' / 'A, B' / ['A'] -> ['A','B','C'] without placeholders."""
    if is_null(v):
        return []
    if isinstance(v, (list, tuple)):
        items = list(v)
    else:
        items = re.split(r"[;,\n]\s*", str(v))
    out = []
    for it in items:
        it = str(it).strip()
        if it and it not in ("0", "None", "none"):
            out.append(it)
    return out
