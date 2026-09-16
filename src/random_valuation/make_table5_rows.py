import csv
from pathlib import Path
R=Path(__file__).resolve().parents[2]/"artifacts"/"random_valuation"
rows=list(csv.DictReader(open(R/"results/experiment_zfix.csv")))
ca_csv=Path(__file__).resolve().parents[2]/"results"/"random_valuation"/"cpbsd_a_rerun.csv"
ca_rows=list(csv.DictReader(open(ca_csv)))
def mean(x): return sum(x)/len(x)
def agg(method,scale,cost,field):
    src = ca_rows if method == "CPBSD-A" else rows
    v=[float(r[field]) for r in src if r["method"]==method and r["variant"]=="fixed" and r["scale"]==scale and r["cost"]==cost and r[field] not in("","None")]
    return mean(v)
# Published Table 5: InS and OOS computed from archived results; Time from original runs.
PUB_TIME={("N10_K50","zero"):{"FCP":0.084,"BSP":0.541,"CPBSD-A":1.828},
          ("N10_K50","random_ind"):{"FCP":1.054,"BSP":1.528,"CPBSD-A":5.742},
          ("N10_K50","random_corr"):{"FCP":1.191,"BSP":1.365,"CPBSD-A":12.899},
          ("N30_K50","zero"):{"FCP":0.114,"BSP":1.367,"CPBSD-A":300.147},
          ("N30_K50","random_ind"):{"FCP":2.860,"BSP":35.186,"CPBSD-A":300.141},
          ("N30_K50","random_corr"):{"FCP":4.686,"BSP":5.149,"CPBSD-A":300.162}}
SC=[("N10_K50","N=10,K=50"),("N30_K50","N=30,K=50")]
COST=[("zero","zero"),("random_ind","random\\_ind"),("random_corr","random\\_corr")]
def b(val,best):
    s="%.3f"%val
    return "\\textbf{%s}"%s if abs(val-best)<5e-4 else s
out=[]
for si,(sc,sclab) in enumerate(SC):
    for ci,(co,colab) in enumerate(COST):
        t=PUB_TIME[(sc,co)]
        fcp=(agg("FCP",sc,co,"ins"),agg("FCP",sc,co,"oos"),t["FCP"])
        bsp=(agg("BSP",sc,co,"ins"),agg("BSP",sc,co,"oos"),t["BSP"])
        ca=(agg("CPBSD-A",sc,co,"ins"),agg("CPBSD-A",sc,co,"oos"),t["CPBSD-A"])
        bins=max(fcp[0],bsp[0],ca[0]); boos=max(fcp[1],bsp[1],ca[1]); btim=min(fcp[2],bsp[2],ca[2])
        def cell(t): return f"{b(t[0],bins)} & {b(t[1],boos)} & {b(t[2],btim)}"
        lead=f"\\({sclab}\\) & \\texttt{{{colab}}}" if ci==0 else f" & \\texttt{{{colab}}}"
        out.append(f"{lead} & {cell(fcp)} & {cell(bsp)} & {cell(ca)} \\\\")
    if si==0: out.append("\\midrule")
print("\n".join(out))
