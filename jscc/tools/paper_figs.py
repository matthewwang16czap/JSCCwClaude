"""The paper's result figures and tables (IEEE TWC), drawn with matplotlib from the results/
files that COMMANDS sections 13-26 wrote. CPU, seconds; no GPU, no simulation.

    python tools/paper_figs.py                         # everything: results/ -> figures/
    python tools/paper_figs.py --only fig05 table02    # some of them
    python tools/paper_figs.py --list                  # what each one shows and reads
    python tools/paper_figs.py --self-test             # synthetic inputs (no results needed)

Each figure and table lists its input files (FIGS, TABLES below). One whose required input is
missing is skipped with a note; optional inputs (CLIC, a second variant) only add series or
groups. Figures are written as figures/<name>.pdf (vector, fonts embedded, for the paper) and
figures/<name>.png (a preview), one IEEE column (3.5 in) or two (7.16 in) wide; tables as
figures/<name>.tex (LaTeX, \\hline rules) and printed as text.

The proposed scheduler is the Lagrangian index; for a quality target its programs use the
calibration-aware reward (schedule_sim.py / rmab_scale.py lagrange_soft, whittle_soft). The
target figures show the indices in that form; the hard-target versions appear in the ablation
(fig05, table02), and elsewhere only where a run had no calibration-aware version (labelled).

The same encoding in every figure (colours checked for colour-vision deficiency on white;
every series also has its own marker and line style, so the figures read in grayscale):
    Lagrangian index       blue      solid, filled; with a hard target: dashed, hollow, hatched
    closed form            orange    the advantage index (mean), the online rule (target)
    Whittle index          aqua      as the Lagrangian index
    planner                yellow    plan_adv (replan_adv where slots are lost)
    PI ratio index         magenta   cross-hatched bars
    PF, PADC               greys     the baselines
    codes, JSCC            green     piecewise-trained / one decoder: solid; a decoder per length:
                                     dashed, hollow; constant-SNR-trained, digital: greys
    true curves (oracle)   ink       dotted
    users, prefix lengths  a blue ramp (dark = the most users, or the longest prefix)
Gains and gaps carry 95% bootstrap intervals over frames, as the tools print them.
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------- the look -------
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, WHITE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN = "#e87ba4", "#008300"
RAMP = ("#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef")         # blue steps, dark -> light
COL1, COL2 = 3.5, 7.16                                                       # IEEE widths, in
FONT = 8.0                                     # IEEE: 8-10 pt type in figures, at printed size
CBRS = ("1/48", "1/24", "1/16", "1/12", "5/48", "1/8")

ENT = {                                                    # entity -> (label, colour, marker)
    "lagrange": ("Lagrangian index", BLUE, "o"),
    "lagrange_hard": ("Lagrangian index, hard target", BLUE, "o"),
    "true": ("hard target, true curves (oracle)", INK, "x"),
    "slope_adv": ("closed form (advantage index)", ORANGE, "s"),
    "online": ("closed form (online rule)", ORANGE, "s"),
    "whittle": ("Whittle index", AQUA, "^"),
    "whittle_hard": ("Whittle index, hard target", AQUA, "^"),
    "planner": ("planner", YELLOW, "D"),
    "pi": ("PI ratio index", MAGENTA, "P"),
    "pf": ("PF", MUTED, "v"),
    "padc": ("PADC", INK2, "X"),
    "jscc": ("JSCC, one decoder", GREEN, "o"),
    "jscc_hd": ("JSCC, a decoder per length", GREEN, "o"),
    "heic": ("HEIC + IR-HARQ", INK2, "s"),
    "gap3": ("3 dB-gap code + IR-HARQ", MUTED, "^"),
    "code": ("piecewise-trained code", GREEN, "o"),
    "code_s1": ("constant-SNR-trained code", MUTED, "d"),
    "code_hd": ("a decoder per length (shared encoder)", GREEN, "o"),
    "percbr": ("one model per CBR", INK2, "h"),
    "kodak": ("Kodak", GREEN, "o"),
    "clic": ("CLIC", INK2, "s"),
}
ALT = ("jscc_hd", "gap3", "code_hd", "lagrange_hard", "whittle_hard")  # dashed, hollow, hatched
PLANNER = {"plan_adv": "planner", "replan_adv": "planner"}
TARGET = dict(PLANNER, lagrange_soft="lagrange", whittle_soft="whittle",   # a target run's
              lagrange="lagrange_hard", whittle="whittle_hard")            # policies -> entities
MEAN_KEYS = ["slope_adv", "pi", "whittle", "lagrange"]
TGT_KEYS = ["online", "planner", "whittle", "whittle_hard", "lagrange", "lagrange_hard"]


def prefer(d):
    """Keep an index's calibration-aware version where it was run, its hard one otherwise."""
    d = dict(d)
    for k in ("whittle", "lagrange"):
        if k in d:
            d.pop(k + "_hard", None)
    return d


def style(key, label=None):
    """Line-plot keywords of an entity."""
    lab, color, marker = ENT[key]
    alt = key in ALT
    kw = dict(color=color, marker=marker, label=label or lab, lw=1.2, ms=4.2, zorder=3,
              ls="--" if alt else "-", mfc=WHITE if alt else color, mec=color, mew=0.9)
    if key == "true":
        kw.update(ls=":", lw=1.1, ms=4.6, mew=1.0)
    return kw


def bar_style(key, label=None):
    lab, color, _ = ENT[key]
    lab = label or lab
    if key in ALT:
        return dict(facecolor=WHITE, edgecolor=color, hatch="////", linewidth=0.7, label=lab)
    if key == "pi":
        return dict(facecolor=color, edgecolor=WHITE, hatch="xxx", linewidth=0.4, label=lab)
    return dict(facecolor=color, edgecolor=WHITE, linewidth=0.4, label=lab)


def ordinal(n):
    """n steps of one blue ramp, dark to light, spread evenly through the palette's steps."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    if n <= 1:
        return [BLUE]
    cmap = LinearSegmentedColormap.from_list("ordinal", RAMP)
    return [to_hex(cmap(t)) for t in ([0.0, 0.7] if n == 2 else np.linspace(0.0, 1.0, n))]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "legend.fontsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 2.0, "figure.facecolor": WHITE,
        "axes.facecolor": WHITE, "savefig.facecolor": WHITE, "pdf.fonttype": 42,
        "ps.fonttype": 42, "errorbar.capsize": 1.6, "hatch.linewidth": 0.6,
        "axes.titlelocation": "left", "axes.titlepad": 3.0, "mathtext.fontset": "custom",
        "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
        "mathtext.cal": "sans:italic", "mathtext.sf": "sans",
    })
    return plt


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".strip(), loc="left", color=INK)


def zero(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(0.0, color=AXIS, lw=0.8, zorder=1)


def k_axis(ax, Ks, label="users K (B = K / 8 resource units per slot)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.minorticks_off()
    ax.set_xlabel(label)


def cbr_axis(ax, label="prefix length: phases (CBR)"):
    ax.set_xticks(range(1, 7))
    ax.set_xticklabels([f"{n}\n{c}" for n, c in zip(range(1, 7), CBRS)])
    ax.set_xlabel(label)


def fig_legend(fig, axes, ncol, extra=(), rows=None):
    """One legend above the figure for every series in `axes` (first label wins). rows: the
    legend as a table, one tuple per row of entity keys or labels (None: an empty cell);
    series it does not name come last. A plain legend loses columns until it fits."""
    from matplotlib.lines import Line2D
    seen = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if not lab.startswith("_"):
                seen.setdefault(lab, h)
    for h in extra:
        seen.setdefault(h.get_label(), h)
    if not seen:
        return
    handles, labels = list(seen.values()), list(seen)
    if rows:
        def lab(k):
            return None if k is None else ENT[k][0] if k in ENT else k
        rows = [r for r in rows if any(lab(k) in seen for k in r)]
        ncol = max(len(r) for r in rows)
        handles, labels = [], []
        for c in range(ncol):                                   # matplotlib fills by column
            for r in rows:
                name = lab(r[c]) if c < len(r) else None
                handles.append(seen[name] if name in seen else Line2D([], [], ls="none"))
                labels.append(name if name in seen else " ")
        rest = [x for x in seen if x not in labels]
        handles += [seen[x] for x in rest]
        labels += rest
    ncol = min(ncol, len(labels))
    while True:
        leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                         ncol=ncol, frameon=False, columnspacing=1.1, handletextpad=0.5,
                         borderaxespad=0.1)
        wide = leg.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
        if rows or ncol <= 1 or wide <= fig.get_figwidth() - 0.04:
            return
        leg.remove()
        ncol -= 1


# --------------------------------------------------------------------------- the inputs -------
class Missing(Exception):
    pass


class Inputs:
    """The files under --results; records which ones a figure read."""

    def __init__(self, root):
        self.root, self.used = root, []

    def path(self, name, required=True):
        p = os.path.join(self.root, name)
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            self.used.append(name)
            return p
        if required:
            raise Missing(name)
        return None

    def json(self, name, required=True):
        p = self.path(name, required)
        if p is None:
            return None
        with open(p) as f:
            return json.load(f)

    def first(self, names):
        """The first of these JSON files that exists, or None."""
        for n in names:
            d = self.json(n, required=False)
            if d is not None:
                return d
        return None


def rmab(d):
    """tools/rmab_scale.py --out: ({K: row}, scale): a target run's values are shares, x 100
    gives points."""
    rows = {int(k): v for k, v in d["per_K"].items()}
    return rows, (1.0 if d["args"].get("target") is None else 100.0)


def rmab_entities(rows, raws, names):
    """{entity: raw policy} of the policies these rows have, renamed by `names`."""
    have = [r for r in raws if any(f"{r}_gap" in row for row in rows.values())]
    return prefer({names.get(r, r): r for r in have})


F = r"[+-]?(?:\d+(?:\.\d+)?|nan|inf)"
ROW_MEAN = re.compile(rf"^(\w+)\s+({F})\s+({F})\s+({F})\s+({F})%\s+({F})\s+({F})\s+({F}) "
                      rf"\[({F}), ({F})\]")
ROW_TGT = re.compile(rf"^(\w+)\s+({F})%\s+({F})\s+({F})%\s+({F})\s+({F})\s+({F})% \[({F}), "
                     rf"({F})\](?:\s+({F})% \[({F}), ({F})\])?")


def parse_sim(path):
    """The printed tables of tools/schedule_sim.py, every run of a file: a list of blocks
    {"set": the header's settings, "kind": "mean" | "target", "base", "rows": {policy: ...},
    "cuts": [slots], "delay": {policy: [% at the target by each cut]}}."""
    blocks, cur, mode = [], None, None
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]
    for ln in lines:
        if re.match(r"^\d+ users, \d+ slots \(", ln):
            cur = {"set": header_settings(ln), "kind": None, "base": None, "rows": {},
                   "cuts": [], "delay": {}}
            blocks.append(cur)
            mode = None
            continue
        if cur is None:
            continue
        if re.match(r"^policy\s+(mean PSNR|satisfied)", ln):
            cur["kind"] = "target" if ln.split()[1] == "satisfied" else "mean"
            b = re.search(r"vs (\w+) \[95% CI\]", ln)
            cur["base"] = b.group(1) if b else None
            mode = "table"
            continue
        if re.match(r"^policy\s+1st", ln):
            cur["cuts"] = [int(x) for x in re.findall(r"by (\d+)", ln)]
            mode = "delay"
            continue
        if mode == "table":
            m = (ROW_TGT if cur["kind"] == "target" else ROW_MEAN).match(ln)
            if m:
                g = m.groups()
                if cur["kind"] == "mean":
                    cur["rows"][g[0]] = {"mean_psnr": float(g[1]), "p5": float(g[2]),
                                         "worst": float(g[3]), "vs": float(g[7]),
                                         "ci": (float(g[8]), float(g[9]))}
                else:
                    r = {"satisfied": float(g[1]) / 100.0, "mean_psnr": float(g[2]),
                         "vs": float(g[6]), "ci": (float(g[7]), float(g[8]))}
                    if g[9] is not None:
                        r["vs_padc"], r["ci_padc"] = float(g[9]), (float(g[10]), float(g[11]))
                    elif cur["base"] == "padc":
                        r["vs_padc"], r["ci_padc"] = r["vs"], r["ci"]
                    cur["rows"][g[0]] = r
        elif mode == "delay" and cur["cuts"]:
            m = re.match(r"^(\w+)\s+(\d+\.\d+)\s+(\d+\.\d+)(.*)$", ln)
            if m:
                by = [float(x) for x in re.findall(r"(\d+\.\d+)%", m.group(4))][:len(cur["cuts"])]
                if len(by) == len(cur["cuts"]):
                    cur["delay"][m.group(1)] = by
    return [b for b in blocks if b["kind"]]


def header_settings(h):
    s = {"users": int(re.match(r"^(\d+) users", h).group(1)),
         "slots": int(re.match(r"^\d+ users, (\d+) slots", h).group(1))}
    m = re.search(r"CQI error ([\d.]+) dB", h)
    s["cqi_error"] = float(m.group(1)) if m else 0.0
    m = re.search(r"stale reports \(correlation ([\d.]+); transmitter (\w+)\)", h)
    s["corr"], s["aware"] = (float(m.group(1)), m.group(2) == "aware") if m else (1.0, None)
    m = re.search(r"([\d.]+)% of slots preempted", h)
    s["preempt"] = float(m.group(1)) / 100.0 if m else 0.0
    m = re.search(r"digital: (\w+) link", h)
    s["link"] = m.group(1) if m else None
    m = re.search(r"HARQ (\w+) \(up to", h)
    s["harq"] = m.group(1) if m else None
    m = re.search(r"utility (\S+)\s*$", h)
    s["utility"] = m.group(1) if m else ""
    return s


def pick(blocks, kind, want):
    """The first block of this kind whose settings match `want`."""
    for b in blocks:
        if b["kind"] == kind and all(
                abs(b["set"].get(k, 0.0) - v) < 1e-9 if isinstance(v, float)
                else b["set"].get(k) == v for k, v in want.items()):
            return b
    return None


def sim_rows(blocks, kind, want, field="vs", base="pf"):
    """{entity: (value, (lo, hi))} of the block matching `want`: the gain over `base` (mean
    runs) or over PADC (field "vs_padc"); a target run's policies renamed (TARGET)."""
    b = pick(blocks, kind, want)
    if b is None or (field == "vs" and b["base"] != base):
        return None
    ci = "ci" if field == "vs" else "ci_padc"
    names = TARGET if kind == "target" else PLANNER
    out = {names.get(p, p): (r[field], r[ci]) for p, r in b["rows"].items() if field in r}
    return prefer(out) if kind == "target" else out


def first_rows(I, files, kind, want, field="vs"):
    """sim_rows of the first of these files that has a matching block (newer runs first)."""
    for fn in files:
        p = I.path(fn, required=False)
        if p:
            r = sim_rows(parse_sim(p), kind, want, field)
            if r:
                return r
    return None


ABS_RULES = (("mean dB", "mean_db"), ("noise variance", "noise"),
             ("capacity (MIESM-like)", "capacity"), ("exponential (EESM)", "eesm"))


def parse_abstractions(path):
    """tools/utility_fit.py --compare-abstractions, every block of a file in order:
    {rule: (MAE with equal weights, MAE with sensitivity weights)}, None where not printed."""
    out, cur = [], None
    with open(path) as f:
        for ln in f:
            if ln.startswith("link abstractions on the held-out profiles"):
                cur = {}
                out.append(cur)
                continue
            if cur is None:
                continue
            s = ln.strip()
            key, rest = None, None
            m = re.match(r"MMSE, kappa \S+\s+(.*)$", s)
            if m:
                key, rest = "mmse", m.group(1)
            else:
                for name, k in ABS_RULES:
                    if s.startswith(name + " "):
                        key, rest = k, s[len(name):]
                        break
            if key is None:
                if s and not s.startswith(("averaging rule", "additive model")):
                    cur = None
                continue
            tok, cells = rest.split(), []
            while tok and len(cells) < 2:
                t = tok.pop(0)
                if t == "-":
                    cells.append(None)
                else:
                    cells.append(float(t))
                    tok = tok[2:]                                  # "/ p95"
            cur[key] = tuple(cells + [None] * (2 - len(cells)))
    return [b for b in out if b]


def decoded_frames(spec, metric="psnr", target=None):
    """{policy: {frame: [per-user values]}} of a decoded replay (tools/mixed_snr.py --design
    file), scored as tools/replay_report.py scores it."""
    from tools.replay_report import score
    per = {}
    for e in spec["entries"]:
        if metric == "psnr":
            v = float(score(e) >= target) if target is not None else float(score(e))
        else:
            v = e.get("actual_" + metric)
            if v is None:
                continue
        per.setdefault(e.get("policy", "?"), {}).setdefault(e["frame"], []).append(float(v))
    return per


def paired(spec, base, metric="psnr", target=None, scale=1.0, names=None):
    """{entity: (mean, (lo, hi))}: each policy minus `base` (a policy as the file names it)
    per frame, the bootstrap of tools/replay_report.py. names: the renaming (default: TARGET
    with a target, else PLANNER; {} keeps the file's names)."""
    from tools.schedule_sim import bootstrap
    names = (TARGET if target is not None else PLANNER) if names is None else names
    per = decoded_frames(spec, metric, target)
    if base not in per:
        return {}
    out = {}
    for p, frames in per.items():
        common = sorted(set(frames) & set(per[base]))
        if p == base or len(common) < 2:
            continue
        d = [np.mean(frames[f]) - np.mean(per[base][f]) for f in common]
        lo, hi = bootstrap(d)
        out[names.get(p, p)] = (scale * float(np.mean(d)), (scale * lo, scale * hi))
    return out


def missed_share(spec, target, names=None):
    """{entity: (%, (lo, hi))}: the users the model predicted at the target whose decode fell
    below it (tools/mixed_snr.py's 'missed'), per frame, bootstrap over frames."""
    from tools.replay_report import score
    from tools.schedule_sim import bootstrap
    names = TARGET if names is None else names
    per = {}
    for e in spec["entries"]:
        miss = e.get("predicted_psnr", 0.0) >= target and score(e) < target
        per.setdefault(e.get("policy", "?"), {}).setdefault(e["frame"], []).append(float(miss))
    out = {}
    for p, frames in per.items():
        d = [np.mean(v) for v in frames.values()]
        if len(d) > 1:
            lo, hi = bootstrap(d)
            out[names.get(p, p)] = (100.0 * float(np.mean(d)), (100.0 * lo, 100.0 * hi))
    return out


def policy_stats(spec, target):
    """Per policy (the file's names) of a decoded target replay: users at the target (%),
    missed (%), mean PSNR (dB) and mean LPIPS (None if not decoded with it)."""
    from tools.replay_report import score
    per = {}
    for e in spec["entries"]:
        per.setdefault(e.get("policy", "?"), []).append(e)
    out = {}
    for p, es in per.items():
        q = np.array([score(e) for e in es])
        pre = np.array([e.get("predicted_psnr", 0.0) for e in es])
        lp = [e["actual_lpips"] for e in es if e.get("actual_lpips") is not None]
        out[p] = {"share": 100.0 * float(np.mean(q >= target)),
                  "missed": 100.0 * float(np.mean((pre >= target) & (q < target))),
                  "mean": float(q.mean()), "lpips": float(np.mean(lp)) if lp else None}
    return out


def constant_curves(d):
    """{prefix length (phases): {SNR: mean PSNR over images}} of a tools/mixed_snr.py run."""
    out = {}
    per = d.get("per_phase") or 1
    for lv in d["levels"]:
        L = int(round(lv["units"] / per))
        ps = np.asarray(lv["psnr"], dtype=np.float64)
        for j, (pr, kind) in enumerate(zip(lv["profiles"], lv["kinds"])):
            if kind == "constant":
                out.setdefault(L, {})[float(pr[0])] = float(np.nanmean(ps[:, j]))
    return out


def _f(v):
    return np.nan if v is None else float(v)


# --------------------------------------------------------------------------- drawing aids -----
def bars(ax, groups, keys, ylabel):
    """Grouped bars with 95% intervals: groups = [(label, {entity: (v, (lo, hi))})]."""
    present = [k for k in keys if any(k in g for _, g in groups)]
    n = max(len(present), 1)
    w = min(0.82 / n, 0.2)
    for i, k in enumerate(present):
        xs, vs, el, eh = [], [], [], []
        for gi, (_, g) in enumerate(groups):
            if k in g:
                v, (lo, hi) = g[k]
                xs.append(gi + (i - (n - 1) / 2) * w)
                vs.append(v)
                el.append(max(v - lo, 0.0))
                eh.append(max(hi - v, 0.0))
        ax.bar(xs, vs, width=w * 0.9, zorder=2, **bar_style(k))
        ax.errorbar(xs, vs, yerr=[el, eh], ls="none", ecolor=INK2, elinewidth=0.7, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    zero(ax)
    ax.set_ylabel(ylabel)


def gap_line(ax, rows, Ks, p, sc, kw, log_floor=None):
    """One policy's gap to the bound against K with its 95% interval (p: the rows' name). On a
    log axis (log_floor: its bottom) an interval reaching 0 runs off the bottom."""
    Ks = [k for k in Ks if f"{p}_gap" in rows.get(k, {})]
    if not Ks:
        return
    g = np.array([rows[k][f"{p}_gap"] for k in Ks]) * sc
    ci = np.array([rows[k][f"{p}_gap_ci"] for k in Ks]) * sc
    lo, hi = np.maximum(g - ci[:, 0], 0), np.maximum(ci[:, 1] - g, 0)
    Ks = np.array(Ks)
    if log_floor is not None:
        ok = g > 0
        Ks, g, hi = Ks[ok], g[ok], hi[ok]
        lo = np.where(ci[ok, 0] > 0, lo[ok], g - 0.5 * log_floor)
    ax.errorbar(Ks, g, yerr=np.vstack([lo, hi]), elinewidth=0.7, **kw)


def _k(p):
    """tools/rmab_scale.py's name of our closed form."""
    return "adv" if p == "slope_adv" else p


# ------------------------------------------------------------------------------ figures -------
def fig02(I, plt):
    """The two trainings compared (C1, C12), at CBR 1/8: (a) how much each phase's slot moves
    the PSNR (its SNR from the lowest to the highest grid value, the other phases at the base
    SNR): the decoder weighs its phases, which is why the abstraction weights slots and why the
    scheduler protects the first ones; (b) what piecewise training costs at constant SNR."""
    a = I.json("p2_full_snrc4.json")
    b = I.json("p2_full_s1.json", required=False)
    codes = [("code", a)] + ([("code_s1", b)] if b is not None else [])
    fig, axes = plt.subplots(1, 2 if b is not None else 1,
                             figsize=(COL2 if b is not None else COL1, 2.4), squeeze=False)
    a1 = axes[0, 0]
    cbr = "1/8" if "1/8" in a["summary"] else list(a["summary"])[-1]
    lo, hi = a["grid"][0], a["grid"][-1]
    w = 0.8 / len(codes)
    for i, (k, d) in enumerate(codes):
        rows = d["summary"][cbr]["oat"]
        swing = [r[-1] - r[0] for r in rows]
        xs = np.arange(1, len(rows) + 1) + (i - (len(codes) - 1) / 2) * w
        a1.bar(xs, swing, width=w * 0.92, zorder=2, **bar_style(k))
    a1.set_xticks(range(1, len(a["summary"][cbr]["oat"]) + 1))
    a1.tick_params(axis="x", length=0)
    a1.set_xlabel(f"phase (the other phases at {a['base_snr']:g} dB)")
    a1.set_ylabel(f"PSNR gain, its slot {lo:g} to {hi:g} dB (dB)")
    panel(a1, "a", f"each phase's weight, CBR {cbr}")
    if b is not None:
        a2 = axes[0, 1]
        L = max(set(constant_curves(a)) & set(constant_curves(b)))
        ca, cb = constant_curves(a)[L], constant_curves(b)[L]
        snrs = sorted(set(ca) & set(cb))
        zero(a2)
        a2.plot(snrs, [ca[x] - cb[x] for x in snrs], color=GREEN, marker="o", ms=4.2, lw=1.2,
                zorder=3, label="_cost")
        a2.set_xticks(snrs)
        a2.set_xlabel("SNR of every slot (dB)")
        a2.set_ylabel("PSNR, piecewise minus\nconstant-SNR-trained (dB)")
        panel(a2, "b", f"the cost of piecewise training, CBR {CBRS[L - 1]}")
    fig_legend(fig, axes, 2)
    return fig


def fig03(I, plt):
    """The abstraction as a link-to-system model (C2): the model's error on every decoded
    history (every policy's schedules), (a) along the predicted PSNR: median and 10-90% range
    per 1 dB; (b) its distribution."""
    sets = (("kodak", ("s22_dec_mean.json", "s22_dec_tgt.json", "s22_dec_r80_mean.json",
                       "s22_dec_r80_tgt.json", "s24_dec_auto.json", "s24_dec_r80.json")),
            ("clic", ("s22_clic_dec_mean.json", "s22_clic_dec_tgt.json", "s24_dec_clic.json")))
    data = []
    for key, files in sets:
        pts = []
        for fn in files:
            d = I.json(fn, required=False)
            if d is not None:
                pts += [(e["predicted_psnr"], e["actual_psnr"]) for e in d["entries"]
                        if e.get("actual_psnr") is not None and "predicted_psnr" in e]
        if pts:
            data.append((key, np.asarray(pts, dtype=np.float64)))
    if not data:
        raise Missing("s22_dec_mean.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    for key, pts in data:
        pred, err = pts[:, 0], pts[:, 0] - pts[:, 1]
        edges = np.arange(np.floor(pred.min()), np.ceil(pred.max()) + 1.0, 1.0)
        idx = np.digitize(pred, edges) - 1
        xs, med, p10, p90 = [], [], [], []
        for j in range(len(edges) - 1):
            e = err[idx == j]
            if len(e) >= max(30, 0.002 * len(err)):
                xs.append(edges[j] + 0.5)
                med.append(np.median(e))
                p10.append(np.percentile(e, 10))
                p90.append(np.percentile(e, 90))
        col = ENT[key][1]
        a1.fill_between(xs, p10, p90, color=col, alpha=0.15, lw=0, zorder=1)
        a1.plot(xs, med, **dict(style(key, f"{ENT[key][0]} ({len(err)} histories)"), ms=3.2))
    zero(a1)
    a1.set_xlabel("predicted PSNR (dB)")
    a1.set_ylabel("predicted minus decoded PSNR (dB)")
    panel(a1, "a", "along the PSNR: median, 10-90% range")
    width = 0.025
    edges = np.arange(-0.5, 0.5 + 1e-9, width)
    notes, top = [], 0.0
    for key, pts in data:
        err = pts[:, 0] - pts[:, 1]
        inside = np.abs(err) <= 0.5
        h = a2.hist(err[inside], bins=edges, histtype="step", color=ENT[key][1], lw=1.1,
                    weights=np.full(inside.sum(), 1.0 / (len(err) * width)))[0]
        top = max(top, float(np.max(h)))
        notes.append(f"{ENT[key][0]}: bias {err.mean():+.3f}, MAE {np.abs(err).mean():.3f} dB; "
                     f"{100 * (1 - inside.mean()):.1f}% beyond 0.5 dB")
    zero(a2, horizontal=False)
    a2.set_ylim(0, 1.45 * top)                                     # room for the numbers
    a2.text(0.02, 0.97, "\n".join(notes), transform=a2.transAxes, va="top", ha="left", color=INK2)
    a2.set_xlabel("predicted minus decoded PSNR (dB)")
    a2.set_ylabel("density")
    panel(a2, "b", "distribution")
    fig_legend(fig, [a1], 2)
    return fig


def fig04(I, plt):
    """The policies against the Lagrangian bound as the cell grows (C13, C3, C4): each one's
    gap per user, (a) mean PSNR, (b) users at 30 dB (the indices calibration-aware); B = K/8
    resource units per slot."""
    mrows, _ = rmab(I.json("s22_scale_mean.json"))
    tgt = I.first(("s24_soft_auto.json", "s22_gap_ties.json"))
    fig, axes = plt.subplots(1, 2 if tgt is not None else 1,
                             figsize=(COL2 if tgt is not None else COL1, 2.5), squeeze=False)
    a1 = axes[0, 0]
    Ks = sorted(mrows)
    ents = [p for p in ("pf", "slope_adv", "pi", "whittle", "lagrange")
            if any(f"{_k(p)}_gap" in r for r in mrows.values())]
    pos = [v[f"{_k(p)}_gap"] for v in mrows.values() for p in ents
           if v.get(f"{_k(p)}_gap", 0) > 0]
    floor = 0.5 * min(pos) if pos else 1e-3
    for p in ents:
        gap_line(a1, mrows, Ks, _k(p), 1.0, style(p), log_floor=floor)
    a1.set_yscale("log")
    a1.set_ylim(bottom=floor)
    k_axis(a1, Ks)
    a1.set_ylabel("gap to the bound per user (dB)")
    panel(a1, "a", "mean PSNR")
    if tgt is not None:
        a2 = axes[0, 1]
        rows, sc = rmab(tgt)
        ents = rmab_entities(rows, ("pf", "whittle", "lagrange", "whittle_soft", "lagrange_soft"),
                             TARGET)
        zero(a2)
        for e in ("whittle", "whittle_hard", "lagrange", "lagrange_hard"):
            if e in ents:
                gap_line(a2, rows, sorted(rows), ents[e], sc, style(e))
        if "pf" in ents:                          # PF is an order of magnitude away: a note
            g = [sc * r["pf_gap"] for r in rows.values() if "pf_gap" in r]
            a2.text(0.98, 0.97, f"PF: {min(g):.1f}-{max(g):.1f} points (off the axis)",
                    transform=a2.transAxes, ha="right", va="top", color=INK2)
        k_axis(a2, sorted(rows))
        a2.set_ylabel("gap to the bound per user (points)")
        panel(a2, "b", "users at 30 dB")
    fig_legend(fig, axes, 6)
    return fig


def _measured_sigma(I):
    """The mean of the per-length sigma 24.1a measured (its printed line), else 0.04 dB."""
    p = I.path("s24_soft_auto.txt", required=False)
    if p:
        with open(p) as f:
            m = re.search(r"sigma ((?:\d+\.\d+ ?)+) dB, bias", f.read())
        if m:
            return float(np.mean([float(x) for x in m.group(1).split()]))
    return 0.04


def fig05(I, plt):
    """Ablation, the calibration-aware reward (C13): (a) the Lagrangian index's gap to the
    bound at 30 dB deciding on the transmitter's estimated curves with a hard target, the same
    with the calibration-aware reward (proposed), and with a hard target on the true curves
    (an oracle); (b) the calibration-aware index against the curve error its reward assumes
    (0: the hard target), K 8 / 32 / 128, the oracle dotted."""
    rows, sc = rmab(I.json("s24_soft_auto.json"))
    tr = I.json("s23_gap_truth.json", required=False)
    trow, tsc = rmab(tr) if tr else ({}, 100.0)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    Ks = sorted(set(rows) | set(trow))
    zero(a1)
    gap_line(a1, rows, Ks, "lagrange", sc, style("lagrange_hard", "hard target"))
    gap_line(a1, rows, Ks, "lagrange_soft", sc, style("lagrange", "calibration-aware (proposed)"))
    if trow:
        gap_line(a1, trow, Ks, "lagrange", tsc, style("true"))
    k_axis(a1, Ks)
    a1.set_ylabel("gap to the bound per user (points)")
    panel(a1, "a", "against the number of users")
    runs = []
    for s, fn in ((0.05, "s24_soft_s05.json"), (0.1, "s24_soft_s10.json"),
                  (0.2, "s24_soft_s20.json")):
        d = I.json(fn, required=False)
        if d is not None:
            runs.append((s, rmab(d)[0]))
    sig = _measured_sigma(I)
    sel = [k for k in (8, 32, 128) if k in rows] or sorted(rows)[:3]
    zero(a2)
    for k, col, mk in zip(sel, ordinal(len(sel))[::-1], ("^", "s", "D")):
        pts = [(0.0, rows[k]["lagrange_gap"] * sc), (sig, rows[k]["lagrange_soft_gap"] * sc)]
        pts += [(s, r[k]["lagrange_soft_gap"] * sc) for s, r in runs
                if "lagrange_soft_gap" in r.get(k, {})]
        pts.sort()
        a2.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker=mk, ms=4.0, lw=1.2,
                label=f"K = {k}", zorder=3)
        if "lagrange_gap" in trow.get(k, {}):
            a2.axhline(trow[k]["lagrange_gap"] * tsc, color=col, lw=0.9, ls=":", zorder=2)
    a2.axvline(sig, color=AXIS, lw=0.8, ls="--", zorder=1)
    a2.text(sig, 0.98, " measured", transform=a2.get_xaxis_transform(), ha="left", va="top",
            color=INK2)
    ticks = [0.0, 0.05, 0.1, 0.15, 0.2]
    a2.set_xticks(ticks)
    a2.set_xticklabels([f"{t:g}" for t in ticks])
    a2.set_xlabel(r"curve error the reward assumes, $\sigma$ (dB; 0: hard target)")
    a2.set_ylabel("gap to the bound per user (points)")
    panel(a2, "b", r"against the assumed $\sigma$")
    fig_legend(fig, [a1, a2], 3, rows=[
        ("hard target", "calibration-aware (proposed)", "true"),
        (f"K = {sel[0]}", f"K = {sel[1]}" if len(sel) > 1 else None,
         f"K = {sel[2]}" if len(sel) > 2 else None)])
    return fig


VARIANTS = (("hard target, as first run", "s22_gap_ties.json", "lagrange"),
            ("the relaxation's tie rule", "s22_gap_ties.json", "lagrange_relax"),
            ("supply-scaled kink prices", "s22_gap_prices.json", "lagrange"),
            ("LP update every 6 slots", "s22_gap_lpu.json", "lagrange_lpu"),
            ("finer program grids", "s22_gap_grid.json", "lagrange"),
            ("margin 0.25 dB", "s23_gap_m25.json", "lagrange"),
            ("margin 0.5 dB", "s23_gap_m50.json", "lagrange"),
            ("calibration-aware reward", "s24_soft_auto.json", "lagrange_soft"),
            ("true curves (oracle)", "s23_gap_truth.json", "lagrange"))


def fig07(I, plt):
    """What closed the 30 dB gap (owed items 9 and 13): every variant of the Lagrangian index
    tested, its gap to the bound per user at K = 8, 32, 128."""
    found = []
    for label, fn, p in VARIANTS:
        d = I.json(fn, required=False)
        if d is not None:
            rows, sc = rmab(d)
            if any(f"{p}_gap" in r for r in rows.values()):
                found.append((label, rows, sc, p))
    if len(found) < 2:
        raise Missing("s22_gap_ties.json and the other gap runs")
    allK = sorted({k for _, rows, _, _ in found for k in rows})
    Ks = [k for k in (8, 32, 128) if k in allK] or allK[:3]
    fig, ax = plt.subplots(figsize=(COL1, 2.9))
    zero(ax, horizontal=False)
    y = np.arange(len(found))[::-1].astype(float)
    for i, (k, col, mk) in enumerate(zip(Ks, ordinal(len(Ks))[::-1], ("o", "s", "D"))):
        xs, ys, el, eh = [], [], [], []
        for yy, (_, rows, sc, p) in zip(y, found):
            r = rows.get(k, {})
            if f"{p}_gap" in r:
                g, (lo, hi) = r[f"{p}_gap"] * sc, r[f"{p}_gap_ci"]
                xs.append(g)
                ys.append(yy - (i - (len(Ks) - 1) / 2) * 0.24)              # K = 8 on top
                el.append(max(g - lo * sc, 0))
                eh.append(max(hi * sc - g, 0))
        ax.errorbar(xs, ys, xerr=[el, eh], ls="none", marker=mk, ms=3.8, color=col, mec=col,
                    elinewidth=0.7, label=f"K = {k}", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f[0] for f in found])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("gap to the bound per user (points at 30 dB)")
    fig_legend(fig, [ax], 3)
    return fig


def fig08(I, plt):
    """Mean PSNR on the main frames, K 8, T 24 (C3, C13): every policy's gain over PF, fresh
    and under each impairment."""
    main = parse_sim(I.path("s22_main_mean.txt"))
    groups = [("fresh reports", sim_rows(main, "mean", {"cqi_error": 0.0, "corr": 1.0})),
              ("CQI error 2 dB", sim_rows(main, "mean", {"cqi_error": 2.0})),
              ("reports r = 0.8", sim_rows(main, "mean", {"corr": 0.8})),
              ("25% of slots lost", first_rows(I, ("s21_lag_mean_more.txt",), "mean",
                                               {"preempt": 0.25})),
              ("T = 16 slots", first_rows(I, ("s21_lag_t16.txt",), "mean", {"slots": 16})),
              ("CLIC images", first_rows(I, ("s22_clic.txt",), "mean",
                                         {"cqi_error": 0.0, "corr": 1.0}))]
    groups = [(n, g) for n, g in groups if g]
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    bars(ax, groups, MEAN_KEYS, "mean PSNR gain over PF (dB)")
    fig_legend(fig, [ax], 4)
    return fig


def fig09(I, plt):
    """A 30 dB target on the main frames (C4, C7): the share of users at 30 dB, every policy's
    gain over PADC, fresh and under each impairment (the indices calibration-aware)."""
    soft = parse_sim(I.path("s24_main_soft.txt"))
    tv = "vs_padc"
    groups = [("fresh reports", sim_rows(soft, "target", {"cqi_error": 0.0, "corr": 1.0}, tv)),
              ("CQI error 2 dB", sim_rows(soft, "target", {"cqi_error": 2.0}, tv)),
              ("reports r = 0.8", first_rows(I, ("s24_main_soft_r80.txt",), "target",
                                             {"corr": 0.8}, tv)),
              ("25% of slots lost", first_rows(I, ("s26_tgt_lost.txt", "s21_lag_tgt_more.txt"),
                                               "target", {"preempt": 0.25}, tv)),
              ("T = 16 slots", first_rows(I, ("s26_tgt_t16.txt", "s21_lag_t16.txt"), "target",
                                          {"slots": 16}, tv)),
              ("CLIC images", first_rows(I, ("s26_tgt_clic.txt", "s22_clic.txt"), "target",
                                         {"cqi_error": 0.0, "corr": 1.0}, tv))]
    groups = [(n, {k: v for k, v in g.items() if k != "padc"}) for n, g in groups if g]
    fig, ax = plt.subplots(figsize=(COL2, 2.4))
    bars(ax, groups, TGT_KEYS, "users at 30 dB: gain over PADC (points)")
    fig_legend(fig, [ax], 6)
    return fig


def fig10(I, plt):
    """A 30 dB target, decoded by the codec (C4, C13): (a) the gain in users at 30 dB over
    PADC, paired by frame; (b) the users the model put at 30 dB that decoded below it. 60
    frames of each policy's schedules; the indices calibration-aware."""
    gains, missed = [], []
    for n, fn in (("Kodak", "s24_dec_auto.json"), ("Kodak, r = 0.8", "s24_dec_r80.json"),
                  ("CLIC", "s24_dec_clic.json")):
        d = I.json(fn, required=False)
        if d is not None:
            tgt = d.get("target") or 30.0
            gains.append((n, prefer(paired(d, "padc", target=tgt, scale=100.0))))
            missed.append((n, prefer(missed_share(d, tgt))))
    if not any(g for _, g in gains):
        raise Missing("s24_dec_auto.json")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL2, 2.5))
    bars(a1, [g for g in gains if g[1]], TGT_KEYS, "gain over PADC (points)")
    panel(a1, "a", "users at 30 dB, decoded")
    bars(a2, missed, ["padc"] + TGT_KEYS, "share of users (%)")
    a2.set_ylim(bottom=0)
    panel(a2, "b", "predicted at 30 dB, decoded below it")
    fig_legend(fig, [a2, a1], 5)
    return fig


def fig11(I, plt):
    """Mean PSNR, decoded by the codec (C3); perceptual metrics follow PSNR: every policy's
    gain over PF in PSNR, MS-SSIM and LPIPS, paired by frame."""
    data = []
    for n, fn in (("Kodak", "s22_dec_mean.json"), ("Kodak, r = 0.8", "s22_dec_r80_mean.json"),
                  ("CLIC", "s22_clic_dec_mean.json")):
        d = I.json(fn, required=False)
        if d is not None:
            data.append((n, d))
    if not data:
        raise Missing("s22_dec_mean.json")
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))
    for ax, (m, title), letter in zip(axes, (("psnr", "PSNR (dB)"), ("msssim", "MS-SSIM"),
                                             ("lpips", "LPIPS (lower is better)")), "abc"):
        groups = [(n, paired(d, "pf", metric=m)) for n, d in data]
        groups = [g for g in groups if g[1]]
        if groups:
            bars(ax, groups, MEAN_KEYS, "")
        panel(ax, letter, title)
    axes[0].set_ylabel("decoded, minus PF")
    fig_legend(fig, axes, 4)
    return fig


def fig12(I, plt):
    """The latency budget (C14): quality against the frame length, K 8; the arrow: how many
    slots the baseline needs for the Lagrangian index's quality at T 24."""
    a = I.json("s21_latency_mean.json", required=False)
    b = I.first(("s26_latency_tgt.json", "s21_latency_tgt.json"))
    if a is None and b is None:
        raise Missing("s21_latency_mean.json")
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), squeeze=False)
    for ax, d, letter, ylab, keys, sc, ref, names in (
            (axes[0, 0], a, "a", "mean PSNR (dB)", ("pf", "slope_adv", "whittle", "lagrange"),
             1.0, "pf", PLANNER),
            (axes[0, 1], b, "b", "users at 30 dB (%)", ("padc", "online", "whittle",
                                                        "whittle_hard", "lagrange",
                                                        "lagrange_hard"), 100.0, "padc", TARGET)):
        if d is None:
            ax.set_visible(False)
            continue
        Ts = d["slots"]
        curves = {names.get(p, p): v for p, v in d["curves"].items()}
        curves = prefer(curves) if names is TARGET else curves
        for k in keys:
            if k in curves:
                ax.plot(Ts, sc * np.asarray(curves[k], dtype=np.float64), **style(k))
        m = d.get("match", {}).get("24")
        r = (m or {}).get("rows", {}).get(ref)
        if r and r.get("flag") in ("", ">"):
            lv = sc * m["level"]                  # every curve is below it left of T = 24
            end = r["needed"] if not r["flag"] else Ts[-1]
            ax.annotate("", xy=(end, lv), xytext=(24, lv),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8, shrinkA=0,
                                        shrinkB=0, ls="-" if not r["flag"] else "--"))
            more = f"+{r['needed'] - 24:.1f}" if not r["flag"] else f"more than +{Ts[-1] - 24}"
            ax.text(23.5, lv, f"{ENT[ref][0]} needs {more} slots", ha="right", va="bottom",
                    color=INK2)
        ax.set_xticks(Ts)
        ax.set_xlabel("frame length T (slots)")
        ax.set_ylabel(ylab)
        panel(ax, letter, "mean PSNR" if letter == "a" else "a 30 dB target")
    fig_legend(fig, axes, 6)
    return fig


def fig13(I, plt):
    """Time to the target (C14): the share of all users at 30 dB by each slot of the frame
    (fresh reports, K 8, T 24; the indices calibration-aware)."""
    b = next((x for x in parse_sim(I.path("s24_main_soft.txt")) if x["delay"]), None)
    if b is None:
        raise Missing("s24_main_soft.txt (its --delay table)")
    delay = prefer({TARGET.get(p, p): v for p, v in b["delay"].items()})
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    for k in ("padc", "online", "planner", "whittle", "whittle_hard", "lagrange",
              "lagrange_hard"):
        if k in delay:
            ax.plot([0] + b["cuts"], [0.0] + delay[k], **style(k))
    ax.set_xticks([0] + b["cuts"])
    ax.set_xlim(0, b["cuts"][-1] + 0.5)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(f"slot of the frame (T = {b['set']['slots']})")
    ax.set_ylabel("users at 30 dB so far (%)")
    fig_legend(fig, [ax], 2)
    return fig


def _best(blocks, kind, axis, x, pred):
    """The best policy (and transmitter) over the blocks at one point of an axis."""
    key = "mean_psnr" if kind == "mean" else "satisfied"
    best = None
    for b in blocks:
        s = b["set"]
        if b["kind"] != kind or not pred(s):
            continue
        if axis == "corr" and not (abs(s["corr"] - x) < 1e-9 and s["cqi_error"] == 0.0):
            continue
        if axis == "cqi" and not (abs(s["cqi_error"] - x) < 1e-9 and s["corr"] == 1.0):
            continue
        for r in b["rows"].values():
            best = r[key] if best is None else max(best, r[key])
    return best


def fig14(I, plt):
    """JSCC against digital with HARQ (C5): the best policy (and transmitter, naive or
    aware) of each system under CQI errors and outdated reports."""
    single = parse_sim(I.path("s23_c5_single.txt"))
    dig = parse_sim(I.path("s20_harq_mean.txt"))
    dig_t, hd = [], []
    for fn in ("s20_harq_tgt.txt", "s20_harq_tgt_aware.txt", "s20_harq_tgt_cqi.txt"):
        p = I.path(fn, required=False)
        if p:
            dig_t += parse_sim(p)
    for fn in ("s18_stale_jscc.txt", "s13_cqi_jscc.txt"):
        p = I.path(fn, required=False)
        if p:
            hd += [b for b in parse_sim(p) if "utility_hd" in b["set"]["utility"]]

    def ir(link):
        return lambda s: s["harq"] == "ir" and s["link"] == link

    def everything(s):
        return True

    series = (("jscc", single, single, everything), ("jscc_hd", hd, hd, everything),
              ("heic", dig, dig_t, ir("cqi")), ("gap3", dig, dig_t, ir("shannon")))
    R, S = (1.0, 0.95, 0.9, 0.8, 0.6, 0.4, 0.0), (0.0, 1.0, 2.0, 3.0)
    fig, axes = plt.subplots(2, 2, figsize=(COL2, 4.2), sharex="col")
    for col, (axis, xs, xlabel) in enumerate((("cqi", S, "CQI error sd (dB)"),
                                              ("corr", R, "report correlation r"))):
        for row, kind in enumerate(("mean", "target")):
            ax = axes[row, col]
            sc = 1.0 if kind == "mean" else 100.0
            for key, mb, tb, pred in series:
                blocks = mb if kind == "mean" else tb
                pts = [(x, _best(blocks, kind, axis, x, pred)) for x in xs]
                pts = [(x, sc * y) for x, y in pts if y is not None]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], **style(key))
            if axis == "corr":
                ax.set_xlim(1.04, -0.04)
                ax.set_xticks([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
            else:
                ax.set_xticks(S)
            if row == 1:
                ax.set_xlabel(xlabel)
            ax.set_ylabel("mean PSNR (dB)" if kind == "mean" else "users at 30 dB (%)")
            panel(ax, "abcd"[2 * row + col], ("mean PSNR" if kind == "mean" else "a 30 dB target")
                  + (", CQI errors" if axis == "cqi" else ", outdated reports"))
    fig_legend(fig, axes, 4)
    return fig


def fig15(I, plt):
    """The scheduler makes the code's channel (C8, C11): how much better than the user's mean
    SNR the first three phases' slots are, against report age and the number of users."""
    from tools.codesign import stats
    by_r = ("bank_sa.json", "bank_sa_cqi95.json", "bank_sa_cqi80.json", "bank_sa_cqi60.json",
            "bank_sa_cqi30.json", "bank_sa_cqi0.json")
    by_k = ("bank_lift_k2.json", "bank_lift_k4.json", "bank_sa.json", "bank_sa_k16.json",
            "bank_lift_k24.json")
    I.path("bank_sa.json")
    cache = {}

    def lifts(fn):
        if fn not in cache:
            d = I.json(fn, required=False)
            cache[fn] = None if d is None else (d, [p["lift"] for p in stats(d)["phase"]])
        return cache[fn]

    ra = sorted(((v[0].get("cqi_corr", 1.0), v[1]) for v in map(lifts, by_r) if v), reverse=True)
    ka = sorted((v[0]["users"], v[1]) for v in map(lifts, by_k) if v)
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.4), sharey=True)
    for ax, data, letter, xlabel, title in (
            (axes[0], ra, "a", "report correlation r", "against report age (K = 8)"),
            (axes[1], ka, "b", "users K (T = 3K slots)", "against the number of users")):
        for j, col in zip(range(3), ordinal(3)):
            pts = [(x, ph[j]) for x, ph in data if len(ph) > j and ph[j] is not None]
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=3.8,
                        lw=1.2, label=f"phase {j + 1}")
        zero(ax)
        ax.set_xlabel(xlabel)
        panel(ax, letter, title)
    axes[0].set_xlim(1.04, -0.04)
    if ka:
        k_axis(axes[1], [k for k, _ in ka], "users K (T = 3K slots)")
    axes[0].set_ylabel("served SNR minus the user's mean SNR (dB)")
    fig_legend(fig, axes, 3)
    return fig


def fig16(I, plt):
    """Encode once with one decoder (C6): what a decoder per length (behind the same encoder)
    and one model per CBR add over the single decoder at each prefix length; mean over the
    constant SNRs, whiskers: the smallest and largest."""
    prefix = constant_curves(I.json("p2_full_snrc4.json"))
    out = {}
    for key, files in (("code_hd", ("p4_full_h48.json", "p4_full_h24.json", "p4_full_h16.json",
                                    "p4_full_h12.json", "p4_full_h5-48.json", "p4_full_h8.json")),
                       ("percbr", tuple(f"p3_full_fix{n}.json" for n in range(1, 7)))):
        cur = {}
        for fn in files:
            d = I.json(fn, required=False)
            if d is not None:
                cur.update(constant_curves(d))
        if cur:
            out[key] = cur
    if not out:
        raise Missing("p4_full_h48.json ... or p3_full_fix1.json ...")
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    zero(ax)
    for key, cur in out.items():
        Ls = sorted(L for L in cur if L in prefix)
        mean, el, eh = [], [], []
        for L in Ls:
            d = [cur[L][s] - prefix[L][s] for s in cur[L] if s in prefix[L]]
            mean.append(float(np.mean(d)))
            el.append(mean[-1] - float(np.min(d)))
            eh.append(float(np.max(d)) - mean[-1])
        ax.errorbar(Ls, mean, yerr=[el, eh], elinewidth=0.7, **style(key))
    cbr_axis(ax)
    ax.set_ylabel("PSNR over the one-decoder code (dB)")
    fig_legend(fig, [ax], 1)
    return fig


# ------------------------------------------------------------------------------- tables -------
RULE_ROWS = (("MIESM (capacity, equal weights)", "capacity", 0), ("capacity", "capacity", 1),
             ("mean dB", "mean_db", 1), ("EESM", "eesm", 1), ("noise variance", "noise", 1),
             ("MMSE (ours)", "mmse", 1))


def _text_table(head, rows):
    cols = list(zip(*([head] + rows)))
    w = [max(len(str(c)) for c in col) for col in cols]
    fmt = lambda r: "  ".join(str(c).rjust(n) if i else str(c).ljust(n)   # noqa: E731
                              for i, (c, n) in enumerate(zip(r, w)))
    return "\n".join([fmt(head)] + [fmt(r) for r in rows])


def table01(I):
    """Link abstraction (C12): the held-out MAE of each effective-SNR rule, four codes on
    Kodak and two on CLIC (replaces a figure)."""
    kod = parse_abstractions(I.path("s13_abstractions.txt"))
    p = I.path("s22_clic.txt", required=False)
    clic = parse_abstractions(p) if p else []
    cols = [(s, tag, blocks[i]) for s, tag, blocks, i in (
        ("Kodak", "(i)", kod, 0), ("Kodak", "(ii)", kod, 2), ("Kodak", "(iii)", kod, 3),
        ("Kodak", "(iv)", kod, 1), ("CLIC", "(i)", clic, 0), ("CLIC", "(iv)", clic, 1))
        if i < len(blocks)]
    if not cols:
        raise Missing("s13_abstractions.txt (no abstraction table in it)")
    vals = [[_f(c[2].get(key, (None, None))[w]) for c in cols] for _, key, w in RULE_ROWS]
    best = [np.nanmin([v[j] for v in vals]) for j in range(len(cols))]
    sets = list(dict.fromkeys(c[0] for c in cols))
    tex = ["% tools/paper_figs.py table01: results/s13_abstractions.txt, s22_clic.txt",
           r"\begin{table}[t]", r"\centering",
           r"\caption{Held-out error (MAE, dB) of effective-SNR link abstractions: each delivery "
           r"history mapped to one SNR on the image's own constant-SNR curve, with sensitivity "
           r"weights except MIESM. Codes: (i) piecewise-trained, one decoder; (ii) "
           r"piecewise-trained, a decoder per length; (iii) one model per CBR; (iv) "
           r"constant-SNR-trained. Bold: the best rule per code.}",
           r"\label{tab:abstraction}",
           r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\hline",
           " & " + " & ".join(rf"\multicolumn{{{sum(c[0] == s for c in cols)}}}{{c}}{{{s}}}"
                              for s in sets) + r" \\",
           "Rule & " + " & ".join(c[1] for c in cols) + r" \\", r"\hline"]
    for (name, _, _), v in zip(RULE_ROWS, vals):
        cells = [("-" if np.isnan(x) else rf"\textbf{{{x:.3f}}}" if abs(x - b) < 1e-12
                  else f"{x:.3f}") for x, b in zip(v, best)]
        tex.append(name + " & " + " & ".join(cells) + r" \\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    head = ["rule"] + [f"{c[0]} {c[1]}" for c in cols]
    rows = [[name] + ["-" if np.isnan(x) else f"{x:.3f}" for x in v]
            for (name, _, _), v in zip(RULE_ROWS, vals)]
    text = ("held-out MAE (dB); (i) piecewise-trained, one decoder; (ii) a decoder per length; "
            "(iii) one model per CBR; (iv) constant-SNR-trained\n" + _text_table(head, rows))
    return "\n".join(tex) + "\n", text


def table02(I):
    """Ablation, the calibration-aware reward, decoded (C13): the Lagrangian index with a hard
    target and with the calibration-aware reward on the same frames: users at 30 dB, the paired
    gain, the users the model put at 30 dB that decoded below it, mean PSNR and LPIPS."""
    rows = []
    for name, tex_name, fn in (("Kodak", "Kodak", "s24_dec_auto.json"),
                               ("Kodak, sigma 0.15 dB", r"Kodak, $\sigma = 0.15$ dB",
                                "s24_dec_s15.json"),
                               ("CLIC", "CLIC", "s24_dec_clic.json"),
                               ("Kodak, r = 0.8", r"Kodak, $r = 0.8$", "s24_dec_r80.json")):
        d = I.json(fn, required=False)
        if d is None:
            continue
        tgt = d.get("target") or 30.0
        st = policy_stats(d, tgt)
        if "lagrange" not in st or "lagrange_soft" not in st:
            continue
        gain = paired(d, "lagrange", target=tgt, scale=100.0, names={}).get("lagrange_soft")
        rows.append((name, tex_name, st["lagrange"], st["lagrange_soft"], gain))
    if not rows:
        raise Missing("s24_dec_auto.json")

    def lp(x):
        return "-" if x is None else f"{x:.3f}"

    def g(x, tex=False):
        if x is None:
            return "-"
        v, (lo, hi) = x
        return (f"${v:+.2f}$ [{lo:.2f}, {hi:.2f}]" if tex else f"{v:+.2f} [{lo:.2f}, {hi:.2f}]")

    tex = ["% tools/paper_figs.py table02: results/s24_dec_*.json", r"\begin{table*}[t]",
           r"\centering",
           r"\caption{Ablation of the calibration-aware reward, decoded by the codec (60 frames "
           r"of each schedule, 30 dB target): the Lagrangian index with a hard target and with "
           r"the calibration-aware reward (proposed) on the same frames. Gain: paired by frame, "
           r"95\% bootstrap interval. Missed: users the model put at 30 dB that decoded below "
           r"it.}",
           r"\label{tab:calibration}", r"\begin{tabular}{lccccccccc}", r"\hline",
           r" & \multicolumn{2}{c}{Users at 30 dB (\%)} & Gain (points) & "
           r"\multicolumn{2}{c}{Missed (\%)} & \multicolumn{2}{c}{Mean PSNR (dB)} & "
           r"\multicolumn{2}{c}{LPIPS} \\",
           r"Schedules & hard & aware & aware $-$ hard & hard & aware & hard & aware & hard & "
           r"aware \\", r"\hline"]
    text_rows = []
    for name, tex_name, h, s, gain in rows:
        tex.append(f"{tex_name} & {h['share']:.1f} & {s['share']:.1f} & {g(gain, True)} & "
                   f"{h['missed']:.1f} & {s['missed']:.1f} & {h['mean']:.2f} & {s['mean']:.2f} & "
                   f"{lp(h['lpips'])} & {lp(s['lpips'])}" + r" \\")
        text_rows.append([name, f"{h['share']:.1f}", f"{s['share']:.1f}", g(gain),
                          f"{h['missed']:.1f}", f"{s['missed']:.1f}", f"{h['mean']:.2f}",
                          f"{s['mean']:.2f}", lp(h["lpips"]), lp(s["lpips"])])
    tex += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    head = ["schedules", "30dB hard", "aware", "gain [95% CI]", "missed hard", "aware",
            "PSNR hard", "aware", "LPIPS hard", "aware"]
    return "\n".join(tex) + "\n", _text_table(head, text_rows)


FIGS = (
    ("fig02_codes", fig02, "the two trainings: how much each phase's slot matters; the cost of "
     "piecewise training at constant SNR", "p2_full_snrc4.json [p2_full_s1.json]"),
    ("fig03_model_error", fig03, "the model's error on every decoded history: along the PSNR, "
     "and its distribution", "s22_dec_*.json s24_dec_*.json [s22_clic_dec_*.json "
     "s24_dec_clic.json]"),
    ("fig04_scaling", fig04, "gap to the bound per user vs K: (a) mean PSNR, (b) users at 30 dB",
     "s22_scale_mean.json [s24_soft_auto.json or s22_gap_ties.json]"),
    ("fig05_ablation_calibration", fig05, "ablation: hard target vs calibration-aware vs true "
     "curves against K; the assumed sigma", "s24_soft_auto.json [s23_gap_truth.json "
     "s24_soft_s05/s10/s20.json s24_soft_auto.txt]"),
    ("fig07_target_variants", fig07, "30 dB: every variant tried on the gap, K 8 / 32 / 128",
     "two or more of s22_gap_ties/prices/lpu/grid.json s23_gap_m25/m50/truth.json "
     "s24_soft_auto.json"),
    ("fig08_main_mean", fig08, "main frames: mean PSNR gain over PF per impairment",
     "s22_main_mean.txt [s21_lag_mean_more.txt s21_lag_t16.txt s22_clic.txt]"),
    ("fig09_main_target", fig09, "main frames: gain in users at 30 dB over PADC per impairment",
     "s24_main_soft.txt [s24_main_soft_r80.txt s26_tgt_lost/t16/clic.txt (else "
     "s21_lag_tgt_more.txt s21_lag_t16.txt s22_clic.txt)]"),
    ("fig10_decoded_target", fig10, "30 dB, decoded: gain over PADC; users predicted at 30 dB "
     "that decoded below", "s24_dec_auto.json [s24_dec_r80.json s24_dec_clic.json]"),
    ("fig11_decoded_mean", fig11, "mean, decoded: gains over PF in PSNR, MS-SSIM and LPIPS",
     "s22_dec_mean.json [s22_dec_r80_mean.json s22_clic_dec_mean.json]"),
    ("fig12_latency", fig12, "quality vs frame length: the latency budget",
     "s21_latency_mean.json and/or s26_latency_tgt.json (else s21_latency_tgt.json)"),
    ("fig13_time_to_target", fig13, "share of users at 30 dB by each slot",
     "s24_main_soft.txt"),
    ("fig14_digital", fig14, "JSCC vs digital with IR-HARQ, CQI errors and outdated reports",
     "s23_c5_single.txt s20_harq_mean.txt [s20_harq_tgt*.txt s18_stale_jscc.txt "
     "s13_cqi_jscc.txt]"),
    ("fig15_scheduler_channel", fig15, "SNR lift of the first phases vs report age and K",
     "bank_sa.json [bank_sa_cqi*.json bank_sa_k16.json bank_lift_k*.json]"),
    ("fig16_decoders", fig16, "a decoder per length and one model per CBR over the single "
     "decoder, by length", "p2_full_snrc4.json and p4_full_h*.json and/or p3_full_fix*.json"),
)
TABLES = (
    ("table01_abstraction", table01, "held-out MAE of each effective-SNR rule per code",
     "s13_abstractions.txt [s22_clic.txt]"),
    ("table02_calibration", table02, "ablation, decoded: hard target vs calibration-aware",
     "s24_dec_auto.json [s24_dec_s15.json s24_dec_clic.json s24_dec_r80.json]"),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default="results", help="the folder the COMMANDS runs wrote")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", nargs="+", default=None,
                    help="figure / table names or prefixes (fig05 table01 ...)")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    ap.add_argument("--dpi", type=int, default=300, help="PNG (and rasterised scatter) resolution")
    ap.add_argument("--list", action="store_true", help="list the figures, tables and inputs")
    ap.add_argument("--self-test", action="store_true",
                    help="draw everything from synthetic inputs in the real tools' formats "
                         "(tools/paper_figs_selftest.py; ~40 s, CPU), stamped as synthetic")
    a = ap.parse_args()
    if a.list:
        for name, _, what, files in FIGS + TABLES:
            print(f"{name:28s} {what}\n{'':28s} reads {files}")
        return
    tmp = None
    if a.self_test:
        import tempfile
        from tools.paper_figs_selftest import build
        tmp = tempfile.TemporaryDirectory(prefix="paper_figs_selftest_")
        a.results = build(tmp.name)
        print(f"synthetic inputs built in {a.results} (removed at the end)")
    plt = setup()
    os.makedirs(a.out, exist_ok=True)
    done, skipped = [], []

    def wanted(name):
        return not a.only or any(name.startswith(o) for o in a.only)

    for name, fn, what, _ in FIGS:
        if not wanted(name):
            continue
        I = Inputs(a.results)
        try:
            fig = fn(I, plt)
        except Missing as e:
            skipped.append(name)
            print(f"  skip  {name}: needs {e}")
            continue
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()                     # labels settle on the first draw; lay out again
        fig.tight_layout(pad=0.3)
        fig.canvas.draw()
        wide = fig.get_tightbbox(fig.canvas.get_renderer()).width
        if wide > fig.get_figwidth() + 0.05:
            print(f"  note  {name}: {wide:.2f} in wide, more than its {fig.get_figwidth():.2f} "
                  f"in column (a legend or label overflows; it would print smaller)")
        if a.self_test:
            fig.text(0.995, 0.005, "synthetic self-test data", ha="right", va="bottom",
                     fontsize=7, color=MUTED)
        for fmt in a.formats:
            fig.savefig(os.path.join(a.out, f"{name}.{fmt}"), dpi=a.dpi, bbox_inches="tight",
                        pad_inches=0.02)
        plt.close(fig)
        done.append(name)
        print(f"  wrote {name}: {what} (from {', '.join(dict.fromkeys(I.used))})")
    for name, fn, what, _ in TABLES:
        if not wanted(name):
            continue
        I = Inputs(a.results)
        try:
            tex, text = fn(I)
        except Missing as e:
            skipped.append(name)
            print(f"  skip  {name}: needs {e}")
            continue
        with open(os.path.join(a.out, f"{name}.tex"), "w") as f:
            f.write(tex)
        done.append(name)
        print(f"  wrote {name}.tex: {what} (from {', '.join(dict.fromkeys(I.used))})")
        print("    " + text.replace("\n", "\n    "))
    print(f"{len(done)} figures and tables in {a.out}/"
          + (f", {len(skipped)} skipped" if skipped else ""))
    if tmp is not None:
        tmp.cleanup()
        if skipped:
            raise SystemExit("self-test: a figure or table was skipped")


if __name__ == "__main__":
    main()
