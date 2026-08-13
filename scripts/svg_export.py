"""Run a figure script and additionally emit an .svg next to every .png it saves,
without editing the script. Usage: python scripts/svg_export.py <figure_script.py> [args...]"""
import sys, runpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure

_orig = matplotlib.figure.Figure.savefig


def savefig_with_svg(self, fname, *a, **k):
    r = _orig(self, fname, *a, **k)
    try:
        s = str(fname)
        if s.lower().endswith(".png"):
            kk = {key: v for key, v in k.items() if key != "dpi"}
            _orig(self, s[:-4] + ".svg", *a, **kk)
            print("  [svg] " + s[:-4] + ".svg", flush=True)
    except Exception as e:
        print("  [svg skip]", e, flush=True)
    return r


matplotlib.figure.Figure.savefig = savefig_with_svg
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
