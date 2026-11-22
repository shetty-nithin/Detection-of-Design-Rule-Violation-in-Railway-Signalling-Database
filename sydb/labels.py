import re
import pandas as pd
 
EXT_HINT = re.compile(r"(SyCR|SyRS|RTI\b|SP_RULE|customer|SIG_RS)", re.I)
 
 
def read_label_table(path):
    """DataFrame with columns rule, y (1 = NOK, 0 = OK, -1 = unlabelled), comment, ext_hint."""
    df = pd.read_excel(path, header=None, usecols="A:C", names=["rule", "res", "comment"])
    df = df.dropna(subset=["rule"])
    df["rule"] = df["rule"].astype(str).str.strip()
    df = df[~df["rule"].str.lower().isin(["rules", "rule", "rule name"])]
    res = df["res"].astype(str).str.strip().str.upper()
    df["y"] = [1 if r == "NOK" else 0 if r == "OK" else -1 for r in res]
    df["comment"] = df["comment"].astype(str).where(df["comment"].notna(), "")
    df["ext_hint"] = df["comment"].map(lambda s: bool(EXT_HINT.search(s)))
    return df.drop_duplicates("rule", keep="first").reset_index(drop=True)[["rule", "y", "comment", "ext_hint"]]
 
 
def read_labels(path, rules=None):
    """{rule: 1 | 0 | None}. If `rules` is given, unknown names are reported."""
    t = read_label_table(path)
    out = {r: (None if y < 0 else int(y)) for r, y in zip(t["rule"], t["y"])}
    if rules is not None:
        unknown = sorted(set(out) - set(rules))
        if unknown:
            print(f"[labels] {len(unknown)} names in {path} not in the rule list, e.g. {unknown[:3]}")
        out = {r: out.get(r) for r in rules}
    return out
