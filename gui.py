#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seg_platform GUI — English · Clear
  Tkinter, no extra deps, calls cli.py
Run: conda activate linjiatai_4090 && python gui.py
"""
import os, sys, glob, pathlib, queue, threading, subprocess, webbrowser
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = pathlib.Path(__file__).resolve().parent
CLI = ROOT / "cli.py"

BG = "#F8FAFC"
CARD = "#FFFFFF"
BORDER = "#E2E8F0"
ACCENT = "#2563EB"
ACCENT_H = "#1D4ED8"
INK = "#0F172A"
FG = INK
MUTED = "#64748B"
CODE_BG = "#0F172A"
CODE_FG = "#E2E8F0"

# --- Font: use real vector fonts available on Linux, avoid bitmap fallback ---
# Inter / JetBrains Mono are not installed on this OS -> Tk falls back to "fixed" bitmap = pixelated
# Pick first available from fc-list so rendering goes through Xft (antialiased)
def _pick_font(candidates):
    try:
        import tkinter.font as tkfont
        avail = set(f.lower() for f in tkfont.families())
        # tkfont.families() can be incomplete in headless Xvfb; also check fontconfig
        # so we probe Tk creation too
        for name in candidates:
            if name.lower() in avail:
                return name
            # try creating the font - if Tk can resolve it without falling back to "fixed"
            try:
                f = tkfont.Font(family=name, size=10)
                actual = f.actual("family").lower()
                if actual == name.lower() or actual not in ("fixed", "clean", "nil"):
                    return name
            except Exception:
                continue
    except Exception:
        pass
    return candidates[-1]

# resolved once Tk root exists; defaults set here and re-resolved in App.__init__
# seg_platform GUI uses smooth humanist sans - Ubuntu is the most fluent installed (round, open curves)
FONT_UI = "Ubuntu"
FONT_MONO = "Ubuntu Mono"

def _resolve_fonts(root=None):
    global FONT_UI, FONT_MONO
    # priority: Ubuntu most fluent -> DejaVu -> Liberation (all vector + CJK friendly, Xft antialiased)
    ui_candidates = ["Ubuntu", "DejaVu Sans", "Liberation Sans", "Noto Sans", "Arial", "Helvetica"]
    mono_candidates = ["Ubuntu Mono", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", "Courier New", "Courier"]
    FONT_UI = _pick_font(ui_candidates) if root is None else _pick_font(ui_candidates)
    FONT_MONO = _pick_font(mono_candidates) if root is None else _pick_font(mono_candidates)
    # If Tk still only knows bitmap fonts (headless), force Ubuntu anyway - real desktop will render correctly
    if FONT_UI.lower() in ("fixed", "clean", "nil"):
        FONT_UI = "Ubuntu"
    if FONT_MONO.lower() in ("fixed", "clean", "nil"):
        FONT_MONO = "Ubuntu Mono"
    return FONT_UI, FONT_MONO

def _style():
    s = ttk.Style()
    try: s.theme_use("clam")
    except: pass
    s.configure("TFrame", background=BG)
    s.configure("Card.TFrame", background=CARD)
    s.configure("TLabel", background=BG, foreground=INK, font=(FONT_UI, 10))
    s.configure("Card.TLabel", background=CARD, foreground=INK, font=(FONT_UI, 10))
    s.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=(FONT_UI, 9))
    s.configure("Title.TLabel", background=CARD, foreground=INK, font=(FONT_UI, 16, "bold"))
    s.configure("Sub.TLabel", background=CARD, foreground=MUTED, font=(FONT_UI, 9))
    s.configure("Section.TLabel", background=CARD, foreground=INK, font=(FONT_UI, 11, "bold"))
    s.configure("TButton", font=(FONT_UI, 10), padding=(10, 6))
    s.configure("Accent.TButton", background=ACCENT, foreground="white", font=(FONT_UI, 10, "bold"))
    s.map("Accent.TButton", background=[("active", ACCENT_H), ("disabled", "#93C5FD")])
    s.configure("Ghost.TButton", background=CARD, foreground=INK)
    s.configure("TEntry", padding=6)
    s.configure("TCheckbutton", background=CARD, font=(FONT_UI, 10))
    s.configure("TRadiobutton", background=CARD, font=(FONT_UI, 10))
    s.configure("Horizontal.TProgressbar", troughcolor=BORDER, background=ACCENT)

class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        cv = tk.Canvas(self, bg=CARD, highlightthickness=0)
        vs = ttk.Scrollbar(self, orient="vertical", command=cv.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=cv.xview)
        self.inner = ttk.Frame(cv, style="Card.TFrame")
        self.inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win = cv.create_window((0,0), window=self.inner, anchor="nw")
        # sync width only if inner narrower than canvas; allow wider content to scroll horizontally
        def _on_cv_config(e):
            cw = e.width
            iw = self.inner.winfo_reqwidth()
            cv.itemconfig(win, width=max(cw, iw) if iw > cw else cw)
        cv.bind("<Configure>", _on_cv_config)
        cv.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        vs.pack(side="right", fill="y")
        hs.pack(side="bottom", fill="x")
        cv.pack(side="left", fill="both", expand=True)
        # vertical scroll; Shift+wheel = horizontal scroll
        def _on_wheel(e):
            # Shift held -> horizontal
            if getattr(e, "state", 0) & 0x1:  # Shift mask
                if getattr(e, "num", None) == 4:
                    cv.xview_scroll(-3, "units")
                elif getattr(e, "num", None) == 5:
                    cv.xview_scroll(3, "units")
                else:
                    cv.xview_scroll(int(-1*(e.delta/120)), "units")
                return
            if getattr(e, "num", None) == 4:
                cv.yview_scroll(-3, "units")
            elif getattr(e, "num", None) == 5:
                cv.yview_scroll(3, "units")
            else:
                cv.yview_scroll(int(-1*(e.delta/120)), "units")
        def _on_shift_wheel(e):
            # explicit Shift+MouseWheel binding
            if getattr(e, "num", None) == 4:
                cv.xview_scroll(-3, "units")
            elif getattr(e, "num", None) == 5:
                cv.xview_scroll(3, "units")
            else:
                cv.xview_scroll(int(-1*(e.delta/120)), "units")
        def _bind(e=None):
            cv.bind_all("<MouseWheel>", _on_wheel)
            cv.bind_all("<Button-4>", _on_wheel)
            cv.bind_all("<Button-5>", _on_wheel)
            cv.bind_all("<Shift-MouseWheel>", _on_shift_wheel)
            cv.bind_all("<Shift-Button-4>", _on_shift_wheel)
            cv.bind_all("<Shift-Button-5>", _on_shift_wheel)
        def _unbind(e=None):
            cv.unbind_all("<MouseWheel>")
            cv.unbind_all("<Button-4>")
            cv.unbind_all("<Button-5>")
            cv.unbind_all("<Shift-MouseWheel>")
            cv.unbind_all("<Shift-Button-4>")
            cv.unbind_all("<Shift-Button-5>")
        for w in (cv, self.inner):
            w.bind("<Enter>", _bind)
            w.bind("<Leave>", _unbind)
        self.canvas = cv
        self.hs = hs
        self._on_wheel = _on_wheel

# ---------- Custom English directory picker (avoids OS language) ----------
class EnglishDirDialog(tk.Toplevel):
    """Simple English-only directory picker to avoid system locale (Chinese) in native dialog."""
    def __init__(self, parent, title="Select Directory", initialdir=""):
        super().__init__(parent)
        self.title(title)
        self.resizable(True, True)
        self.geometry("640x420")
        self.result = None
        self.transient(parent)
        self.grab_set()
        # path entry
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Path:").pack(side="left")
        self.path_var = tk.StringVar(value=initialdir if initialdir and os.path.isdir(initialdir) else os.path.expanduser("~"))
        if not os.path.isdir(self.path_var.get()):
            self.path_var.set(str(ROOT))
        ent = ttk.Entry(top, textvariable=self.path_var)
        ent.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(top, text="Up", width=6, command=self._go_up).pack(side="left", padx=2)
        ttk.Button(top, text="Go", width=6, command=self._refresh).pack(side="left")
        # list
        mid = ttk.Frame(self, padding=(10,0,10,0))
        mid.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(mid, selectmode="single", font=(FONT_UI, 10), activestyle="dotbox")
        vs = ttk.Scrollbar(mid, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=vs.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.listbox.bind("<Double-Button-1>", lambda e: self._enter())
        # buttons
        bot = ttk.Frame(self, padding=10)
        bot.pack(fill="x")
        ttk.Button(bot, text="Cancel", width=12, command=self._cancel).pack(side="right", padx=4)
        ttk.Button(bot, text="Select", width=12, style="Accent.TButton", command=self._select).pack(side="right", padx=4)
        self.bind("<Return>", lambda e: self._select())
        self.bind("<Escape>", lambda e: self._cancel())
        ent.bind("<Return>", lambda e: self._refresh())
        self._refresh()
        self.wait_window(self)

    def _refresh(self):
        p = self.path_var.get().strip() or str(ROOT)
        if not os.path.isdir(p):
            p = os.path.dirname(p) or str(ROOT)
            self.path_var.set(p)
        self.listbox.delete(0, tk.END)
        try:
            dirs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            dirs.sort(key=str.lower)
            for d in dirs:
                self.listbox.insert(tk.END, d)
        except Exception as e:
            self.listbox.insert(tk.END, f"[error: {e}]")

    def _go_up(self):
        p = self.path_var.get().strip()
        up = os.path.dirname(p.rstrip(os.sep)) or p
        if up and os.path.isdir(up):
            self.path_var.set(up)
            self._refresh()

    def _enter(self):
        sel = self.listbox.curselection()
        if not sel: return
        name = self.listbox.get(sel[0])
        if name.startswith("[error"): return
        nxt = os.path.join(self.path_var.get().strip(), name)
        if os.path.isdir(nxt):
            self.path_var.set(nxt)
            self._refresh()

    def _select(self):
        p = self.path_var.get().strip()
        if os.path.isdir(p):
            self.result = p
            self.destroy()

    def _cancel(self):
        self.result = None
        self.destroy()

def ask_directory_english(parent, title="Select Directory", initialdir=""):
    dlg = EnglishDirDialog(parent, title=title, initialdir=initialdir)
    return dlg.result

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        _resolve_fonts(self)
        self.title("seg_platform — Unified WSI Segmentation")
        self.geometry("1240x780")
        self.minsize(1120, 680)
        self.configure(bg=BG)
        _style()
        self.proc = None
        self.log_q = queue.Queue()
        self.vars = {}
        self._build()
        self._switch_model()
        self.after(100, self._poll)

    def _build(self):
        top = tk.Frame(self, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        top.pack(fill="x", padx=12, pady=(12, 8))
        top.columnconfigure(1, weight=1)
        left = ttk.Frame(top, style="Card.TFrame", padding=(16, 12))
        left.grid(row=0, column=0, sticky="w")
        ttk.Label(left, text="seg_platform", style="Title.TLabel").pack(anchor="w")
        right = ttk.Frame(top, style="Card.TFrame", padding=(0, 12))
        right.grid(row=0, column=1, sticky="e", padx=12)
        ttk.Button(right, text="Docs", style="Ghost.TButton", command=self._open_readme).pack(side="right", padx=4)
        ttk.Button(right, text="Open Folder", style="Ghost.TButton", command=lambda: self._open_folder(str(ROOT))).pack(side="right", padx=4)

        main = ttk.Frame(self)
        main.pack(fill="both", expand=True, padx=12, pady=6)
        main.columnconfigure(0, weight=1, uniform="a")
        main.columnconfigure(1, weight=1, uniform="a")
        main.rowconfigure(0, weight=1)

        left_card = tk.Frame(main, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        left_card.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        sf = ScrollFrame(left_card)
        sf.pack(fill="both", expand=True, padx=1, pady=1)
        self.form = sf.inner

        right_card = tk.Frame(main, bg=CARD, bd=1, relief="solid", highlightbackground=BORDER)
        right_card.grid(row=0, column=1, sticky="nsew", padx=(6,0))
        right_card.rowconfigure(1, weight=1)
        right_card.columnconfigure(0, weight=1)
        hdr = ttk.Frame(right_card, style="Card.TFrame", padding=(12,10))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Log", style="Section.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(hdr, textvariable=self.status_var, style="Card.TLabel", foreground=MUTED).pack(side="right")
        self.pbar = ttk.Progressbar(hdr, mode="indeterminate", style="Horizontal.TProgressbar")
        box = ttk.Frame(right_card, style="Card.TFrame", padding=8)
        box.grid(row=1, column=0, sticky="nsew")
        box.rowconfigure(0, weight=1); box.columnconfigure(0, weight=1)
        self.log_text = tk.Text(box, bg=CODE_BG, fg=CODE_FG, insertbackground="white",
                                font=(FONT_MONO, 9), wrap="word", bd=0, padx=10, pady=10,
                                selectbackground="#334155", selectforeground="white",
                                inactiveselectbackground="#334155", exportselection=True,
                                undo=True)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        vs = ttk.Scrollbar(box, orient="vertical", command=self.log_text.yview)
        vs.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=vs.set)
        # context menu + copy bindings for log
        self._log_menu = tk.Menu(self.log_text, tearoff=0)
        self._log_menu.add_command(label="Copy", command=lambda: self.log_text.event_generate("<<Copy>>"))
        self._log_menu.add_command(label="Copy All", command=self._copy_all_log)
        self._log_menu.add_command(label="Select All", command=lambda: self._select_all_log())
        self._log_menu.add_separator()
        self._log_menu.add_command(label="Clear", command=self._clear)
        self._log_menu.add_command(label="Save to file…", command=self._save_log)
        self.log_text.bind("<Button-3>", lambda e: self._log_menu.tk_popup(e.x_root, e.y_root))
        self.log_text.bind("<Control-c>", lambda e: None)  # allow default
        self.log_text.bind("<Control-a>", lambda e: (self._select_all_log(), "break")[1])
        # hint
        hint = ttk.Label(right_card, text="Tip: Select text and press Ctrl+C to copy, or right-click for menu.", style="Muted.TLabel")
        hint.grid(row=3, column=0, sticky="w", padx=10, pady=(0,4))
        bar = ttk.Frame(right_card, style="Card.TFrame", padding=(10,8))
        bar.grid(row=2, column=0, sticky="ew")
        self.run_btn = ttk.Button(bar, text="▶  Run", style="Accent.TButton", command=self._run)
        self.run_btn.pack(side="left")
        ttk.Button(bar, text="■  Stop", command=self._stop).pack(side="left", padx=6)
        ttk.Button(bar, text="Open Output", command=self._open_output).pack(side="left", padx=6)
        ttk.Button(bar, text="Open QuPath", command=self._preview).pack(side="left", padx=6)
        ttk.Button(bar, text="Clear", command=self._clear).pack(side="right")

        self._add_model()
        self._add_io()
        self._add_common()
        self._add_qupath()
        self._add_cellpose()
        self._add_cerberus()

    # helpers
    def _var(self, k, d=""):
        v = tk.StringVar(value=str(d)); self.vars[k]=v; return v
    def _bool(self, k, d=False):
        v = tk.BooleanVar(value=bool(d)); self.vars[k]=v; return v
    def _section(self, title, subtitle=""):
        outer = ttk.Frame(self.form, style="Card.TFrame", padding=(14,12))
        outer.pack(fill="x", padx=12, pady=8)
        ttk.Label(outer, text=title, style="Section.TLabel").pack(anchor="w")
        if subtitle:
            ttk.Label(outer, text=subtitle, style="Muted.TLabel", wraplength=520, justify="left").pack(anchor="w", pady=(3,6))
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(8,10))
        return outer
    def _field(self, parent, label, var, browse=False, browse_title="Select Directory"):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, style="Card.TLabel", width=22).pack(side="left")
        e = ttk.Entry(row, textvariable=var)
        e.pack(side="left", fill="x", expand=True, padx=(6,6))
        if browse:
            ttk.Button(row, text="Browse…", width=12, command=lambda: self._browse(var, browse_title)).pack(side="left")
        return e
    def _num(self, parent, label, var, hint=""):
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label, style="Card.TLabel", width=22).pack(side="left")
        ttk.Entry(row, textvariable=var, width=18).pack(side="left", padx=6)
        if hint:
            ttk.Label(row, text=hint, style="Card.TLabel", foreground=MUTED).pack(side="left")
        return row

    def _add_model(self):
        sec = self._section("Model")
        self.model_var = tk.StringVar(value="cellpose")
        row = ttk.Frame(sec, style="Card.TFrame"); row.pack(fill="x", pady=2)
        ttk.Radiobutton(row, text="Cellpose (cpsam / cyto) — Fast · Low VRAM", variable=self.model_var, value="cellpose", command=self._switch_model).pack(side="left", padx=6)
        ttk.Radiobutton(row, text="Cerberus (Gland / Lumen / Nuclei) — 4 tasks", variable=self.model_var, value="cerberus", command=self._switch_model).pack(side="left", padx=18)

    def _add_io(self):
        sec = self._section("Input / Output")
        self._field(sec, "Input Directory *", self._var("input_dir"), browse=True, browse_title="Select Input Directory")
        self._field(sec, "Output Directory *", self._var("output_dir"), browse=True, browse_title="Select Output Directory")
        self._field(sec, "Mask Directory", self._var("msk_dir"), browse=True, browse_title="Select Mask Directory")
        r = ttk.Frame(sec, style="Card.TFrame"); r.pack(fill="x", pady=4)
        ttk.Label(r, text="WSI Extension", style="Card.TLabel", width=22).pack(side="left")
        ttk.Entry(r, textvariable=self._var("wsi_file_ext",".svs,.tiff"), width=18).pack(side="left", padx=6)
        ttk.Label(r, text="comma-separated, * for all", style="Card.TLabel", foreground=MUTED).pack(side="left", padx=6)
        # GPU — dedicated row with wider input
        gr = ttk.Frame(sec, style="Card.TFrame"); gr.pack(fill="x", pady=4)
        ttk.Label(gr, text="GPU", style="Card.TLabel", width=22).pack(side="left")
        ttk.Entry(gr, textvariable=self._var("gpu","0"), width=28).pack(side="left", padx=6)
        ttk.Label(gr, text="CUDA_VISIBLE_DEVICES  ·  e.g. 0  /  0,1  /  empty = CPU", style="Card.TLabel", foreground=MUTED).pack(side="left", padx=6)

    def _add_common(self):
        sec = self._section("WSI / Post-processing")
        g = ttk.Frame(sec, style="Card.TFrame"); g.pack(fill="x")
        L = ttk.Frame(g, style="Card.TFrame"); L.pack(side="left", fill="x", expand=True, padx=(0,8))
        R = ttk.Frame(g, style="Card.TFrame"); R.pack(side="left", fill="x", expand=True)
        self._num(L, "wsi_proc_mag (mpp)", self._var("wsi_proc_mag","0.5"), "0.5=20×  0.25=40×")
        self._num(L, "tile_shape (px)", self._var("tile_shape","4096"), "2048 if OOM")
        self._num(L, "ambiguous_size (px)", self._var("ambiguous_size","64"), "")
        self._num(L, "batch_size", self._var("batch_size","30"), "")
        self._num(R, "chunk_shape (px)", self._var("chunk_shape","15000"))
        self._num(R, "patch_input (px)", self._var("patch_input_shape","448"), "")
        self._num(R, "patch_output (px)", self._var("patch_output_shape","144"), "")
        self._num(R, "nr_post_proc_workers", self._var("nr_post_proc_workers","0"))
        row = ttk.Frame(sec, style="Card.TFrame"); row.pack(fill="x", pady=(8,2))
        self._bool("save_thumb", False); self._bool("save_mask", False)
        tk.Checkbutton(row, text="save_thumb  →  thumb/", variable=self.vars["save_thumb"], bg=CARD, activebackground=CARD, selectcolor="white", fg=FG, activeforeground=FG, font=(FONT_UI, 10), highlightthickness=0, bd=0, anchor="w").pack(side="left", padx=4)
        tk.Checkbutton(row, text="save_mask  →  mask/", variable=self.vars["save_mask"], bg=CARD, activebackground=CARD, selectcolor="white", fg=FG, activeforeground=FG, font=(FONT_UI, 10), highlightthickness=0, bd=0, anchor="w").pack(side="left", padx=12)

    def _add_qupath(self):
        sec = self._section("QuPath Export")
        row = ttk.Frame(sec, style="Card.TFrame"); row.pack(fill="x")
        self._bool("save_qupath", False)
        tk.Checkbutton(row, text="Export QuPath GeoJSON  →  qupath/<basename>.geojson", variable=self.vars["save_qupath"], bg=CARD, activebackground=CARD, selectcolor="white", fg=FG, activeforeground=FG, font=(FONT_UI, 10, "bold"), highlightthickness=0, bd=0, anchor="w").pack(side="left")

    def _add_cellpose(self):
        self.cellpose_frame = self._section("Cellpose Parameters", "cpsam / cyto / cyto2 / cyto3 or local path")
        self._field(self.cellpose_frame, "model", self._var("cellpose_model","cpsam"))
        self._num(self.cellpose_frame, "diameter", self._var("diameter",""), "None = auto, 15–20 for 0.5 mpp H&E")
        g = ttk.Frame(self.cellpose_frame, style="Card.TFrame"); g.pack(fill="x")
        self._num(g, "flow_threshold", self._var("flow_threshold","0.4"), "0.4 default; ↑ stricter (fewer merges)")
        self._num(g, "cellprob_threshold", self._var("cellprob_threshold","0.0"), "0.0 default; ↑ fewer cells (higher confidence)")
        self._num(g, "min_size", self._var("min_size","15"), "drop smaller masks")
        row = ttk.Frame(self.cellpose_frame, style="Card.TFrame"); row.pack(fill="x", pady=4)
        self._bool("use_cerberus_infer", False)
        tk.Checkbutton(row, text="use_cerberus_infer  (chunk + flow memmap)", variable=self.vars["use_cerberus_infer"], bg=CARD, activebackground=CARD, selectcolor="white", fg=FG, activeforeground=FG, font=(FONT_UI, 10), highlightthickness=0, bd=0, anchor="w").pack(side="left", padx=4)

    def _add_cerberus(self):
        self.cerberus_frame = self._section("Cerberus Parameters")
        self._field(self.cerberus_frame, "Weights dir", self._var("cerberus_model", str(ROOT/"models/cerberus/pretrained_weights/resnet34_cerberus")), browse=True, browse_title="Select Cerberus Weights Directory")
        self._field(self.cerberus_frame, "Cache path (SSD 100GB+)", self._var("cache_path",""), browse=True, browse_title="Select Cache Directory")
        self._field(self.cerberus_frame, "Logging dir", self._var("logging_dir",""), browse=True, browse_title="Select Logging Directory")
        self._num(self.cerberus_frame, "nr_inference_workers", self._var("nr_inference_workers","0"), "")

    def _switch_model(self):
        if self.model_var.get() == "cellpose":
            self.cerberus_frame.pack_forget()
            self.cellpose_frame.pack(fill="x", padx=12, pady=8)
        else:
            self.cellpose_frame.pack_forget()
            self.cerberus_frame.pack(fill="x", padx=12, pady=8)
    def _browse(self, var, title="Select Directory"):
        init = var.get().strip()
        init_dir = init if init and os.path.isdir(init) else str(ROOT)
        # Use custom English dialog to avoid OS Chinese (选择/确定/取消)
        try:
            d = ask_directory_english(self, title=title, initialdir=init_dir)
        except Exception:
            d = filedialog.askdirectory(title=title, initialdir=init_dir, mustexist=False)
        if d: var.set(d)
    def _open_folder(self, p):
        try:
            if sys.platform.startswith("linux"): subprocess.Popen(["xdg-open", p])
            elif sys.platform == "darwin": subprocess.Popen(["open", p])
            else: os.startfile(p)
        except Exception as e:
            messagebox.showinfo("Open", f"{p}\n{e}")
    def _open_output(self):
        p = self.vars["output_dir"].get().strip()
        if p and os.path.isdir(p): self._open_folder(p)
        else: messagebox.showwarning("Tip", "Output directory not found")
    def _open_readme(self):
        p = ROOT / "README.md"
        if p.exists(): self._open_folder(str(p))
    def _preview(self):
        # Directly launch QuPath; if output/qupath exists also open its folder
        qbin = "/home/linjiatai/QuPath/QuPath/bin/QuPath"
        out = self.vars["output_dir"].get().strip()
        # if output/qupath exists, open folder first
        if out and os.path.isdir(out):
            qp = os.path.join(out, "qupath")
            if os.path.isdir(qp) and os.listdir(qp):
                self._open_folder(qp)
        # launch QuPath app directly
        try:
            if os.path.isfile(qbin) and os.access(qbin, os.X_OK):
                subprocess.Popen([qbin], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                self.status_var.set("QuPath launched")
                return
            # fallback: open bin folder
            qbin_dir = os.path.dirname(qbin)
            if os.path.isdir(qbin_dir):
                self._open_folder(qbin_dir); return
        except Exception as e:
            messagebox.showerror("QuPath", f"Failed to launch QuPath\n{qbin}\n{e}")
            return
        messagebox.showwarning("Tip", f"QuPath not found at {qbin}")
    def _clear(self): self.log_text.delete("1.0", tk.END)
    def _select_all_log(self):
        self.log_text.tag_add("sel", "1.0", "end-1c")
        self.log_text.mark_set("insert", "1.0")
        self.log_text.see("insert")
        return "break"
    def _copy_all_log(self):
        txt = self.log_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(txt)
        self.status_var.set("Copied to clipboard")
        self.after(1500, lambda: self.status_var.set("Ready" if not (self.proc and self.proc.poll() is None) else "Running…"))
    def _save_log(self):
        p = filedialog.asksaveasfilename(title="Save Log", defaultextension=".log", filetypes=[("Log","*.log"),("Text","*.txt"),("All","*.*")], initialdir=str(ROOT))
        if p:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(self.log_text.get("1.0", "end-1c"))
                self.status_var.set(f"Saved to {p}")
            except Exception as e:
                messagebox.showerror("Save failed", str(e))
    def _log(self, s): self.log_q.put(s)
    def _poll(self):
        try:
            while True:
                s = self.log_q.get_nowait()
                self.log_text.insert(tk.END, s); self.log_text.see(tk.END)
        except queue.Empty: pass
        self.after(80, self._poll)
    def _validate(self):
        inp = self.vars["input_dir"].get().strip()
        out = self.vars["output_dir"].get().strip()
        if not inp or not os.path.isdir(inp):
            messagebox.showerror("Error", "Input directory not found"); return False
        if not out:
            messagebox.showerror("Error", "Output directory is required"); return False
        return True
    def _build_cmd(self):
        m = self.model_var.get()
        g = lambda k, d="": (self.vars[k].get().strip() or d)
        args = [sys.executable, str(CLI), "--model", m,
                "--input_dir", g("input_dir"), "--output_dir", g("output_dir"),
                "--wsi_file_ext", g("wsi_file_ext",".svs"), "--gpu", g("gpu","0"),
                "--wsi_proc_mag", g("wsi_proc_mag","0.5"), "--tile_shape", g("tile_shape","4096"),
                "--ambiguous_size", g("ambiguous_size","64"),
                "--patch_input_shape", g("patch_input_shape","448"), "--patch_output_shape", g("patch_output_shape","144"),
                "--chunk_shape", g("chunk_shape","15000"), "--batch_size", g("batch_size","30"),
                "--nr_post_proc_workers", g("nr_post_proc_workers","0")]
        if g("msk_dir"): args += ["--msk_dir", g("msk_dir")]
        if self.vars["save_thumb"].get(): args.append("--save_thumb")
        if self.vars["save_mask"].get(): args.append("--save_mask")
        if self.vars["save_qupath"].get():
            args.append("--save_qupath")
        # cellpose cerberus-style
        try:
            if self.vars["use_cerberus_infer"].get():
                args.append("--use_cerberus_infer")
        except Exception:
            pass
        if m == "cellpose":
            args += ["--cellpose_model", g("cellpose_model","cpsam")]
            if g("diameter"): args += ["--diameter", g("diameter")]
            args += ["--flow_threshold", g("flow_threshold","0.4"), "--cellprob_threshold", g("cellprob_threshold","0.0"), "--min_size", g("min_size","15")]
        else:
            if g("cerberus_model"): args += ["--cerberus_model", g("cerberus_model")]
            if g("cache_path"): args += ["--cache_path", g("cache_path")]
            if g("logging_dir"): args += ["--logging_dir", g("logging_dir")]
            args += ["--nr_inference_workers", g("nr_inference_workers","0")]
        return args
    def _run(self):
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Tip", "Already running"); return
        if not self._validate(): return
        cmd = self._build_cmd()
        pretty = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        self._clear()
        self._log(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] $ {pretty}\n")
        self.status_var.set("Running…"); self.pbar.start(12); self.run_btn.configure(state="disabled")
        def _worker():
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(ROOT)+os.pathsep+env.get("PYTHONPATH","")
                self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                for line in self.proc.stdout:
                    self._log(line)
                self.proc.wait()
                code = self.proc.returncode
                self._log(f"\n{'Done' if code==0 else f'Failed exit {code}'} — {self.vars['output_dir'].get().strip()}\n")
                self.status_var.set("Done" if code==0 else f"Failed {code}")
            except Exception as e:
                self._log(f"\n[GUI ERROR] {e}\n"); self.status_var.set("Error")
            finally:
                self.pbar.stop(); self.run_btn.configure(state="normal"); self.proc=None
        threading.Thread(target=_worker, daemon=True).start()
    def _stop(self):
        if self.proc and self.proc.poll() is None:
            try: self.proc.terminate(); self._log("\n[GUI] Terminated\n"); self.status_var.set("Stopped"); self.pbar.stop(); self.run_btn.configure(state="normal")
            except Exception as e: messagebox.showerror("Error", str(e))
        else: self.status_var.set("Ready")

if __name__ == "__main__":
    App().mainloop()
