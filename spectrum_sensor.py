"""
ENCS5323 Project – Component 1: Real-World Spectrum Sensing
Device : ADALM-PLUTO SDR (RX unit)
Display: Configurable via DISP_START_MHZ / DISP_END_MHZ in code (default 900–2500 MHz).

Install:  pip install pyadi-iio numpy matplotlib scipy
Run:      python spectrum_sensor.py
Demo:     python spectrum_sensor.py --demo   (no hardware; for UI testing)
"""

import argparse
import csv
import os
import threading
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from scipy.signal import windows

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

PLUTO_URI       = "usb:"
SAMPLE_RATE     = 20e6        # 20 MHz instantaneous BW
FFT_SIZE        = 4096
NUM_FRAMES      = 2           # averaged per 20 MHz chunk (lower = faster)
RX_GAIN_MODE    = "manual"      # "manual" gives stronger readings than slow_attack indoors
RX_GAIN_DB      = 50            # 0–73 dB on Pluto (raise if still flat; lower if clipping)
LO_SETTLE_SEC   = 0.05
RX_FLUSH_FRAMES = 3             # discard stale samples after each LO change
UPDATE_INTERVAL = 200         # UI refresh (ms) — sweep runs in background
CONNECT_TIMEOUT = 8.0         # seconds before giving up on Pluto USB

# ══════════════════════════════════════════════════════════════
#  FREQUENCY RANGE — edit here only (no command-line input)
#  Doctor / project: change MHz values, save, re-run script.
# ══════════════════════════════════════════════════════════════
#
#  Default — full assignment range (all four bands visible):
DISP_START_MHZ = 2400
DISP_END_MHZ   = 2500
#
#  Examples (uncomment ONE pair to zoom):
#  DISP_START_MHZ = 935          # GSM 900 downlink only
#  DISP_END_MHZ   = 960
#  
#  DISP_START_MHZ = 2110         # UMTS 2100 downlink only
#  DISP_END_MHZ   = 2170
#
#  DISP_START_MHZ = 2400         # 2.4 GHz ISM only
#  DISP_END_MHZ   = 2500
#
DISP_START = DISP_START_MHZ * 1e6
DISP_END   = DISP_END_MHZ * 1e6
GRID_PTS   = 8192

# Sweep always matches the display range above
SCAN_REGIONS = [(DISP_START, DISP_END)]

# Set True in code for faster sweep (cellular + ISM chunks only, not full span)
USE_BANDS_ONLY_SWEEP = False
SCAN_REGIONS_BANDS_ONLY = [
    (900e6,  970e6),
    (1800e6, 1885e6),
    (2105e6, 2175e6),
    (2395e6, 2505e6),
]

ISM_BAND_LO = 2400e6
ISM_BAND_HI = 2500e6

# ── Band highlights + channel / carrier markers ─────────────────
def _gsm_carriers_mhz(f0_mhz: float, count: int, step_mhz: float = 0.2, every: int = 1):
    return [(f0_mhz * 1e6 + i * step_mhz * 1e6, "") for i in range(0, count, every)]

BANDS = [
    {
        "label": "GSM 900 Downlink",
        "f_start": 935e6, "f_end": 960e6,
        "shade": "#00e5ff", "shade_alpha": 0.18,
        "markers": _gsm_carriers_mhz(935.2, 124, every=10),
        "markers_zoom_every": 2,
    },
    {
        "label": "GSM 1800 Downlink",
        "f_start": 1805e6, "f_end": 1880e6,
        "shade": "#69ff47", "shade_alpha": 0.18,
        "markers": _gsm_carriers_mhz(1805.2, 374, every=25),
        "markers_zoom_every": 5,
    },
    {
        "label": "UMTS 2100 Downlink",
        "f_start": 2110e6, "f_end": 2170e6,
        "shade": "#ff9f47", "shade_alpha": 0.20,
        "markers": [(2112.4e6 + i * 5e6, f"F{i+1}") for i in range(12)],
        "markers_zoom_every": 1,
    },
    {
        "label": "2.4 GHz ISM (Wi-Fi / BT / ZigBee)",
        "f_start": 2400e6, "f_end": 2500e6,
        "shade": "#ff4f7b", "shade_alpha": 0.18,
        "markers": [
            (2412e6, "CH1"), (2417e6, "CH2"), (2422e6, "CH3"), (2427e6, "CH4"),
            (2432e6, "CH5"), (2437e6, "CH6"), (2442e6, "CH7"), (2447e6, "CH8"),
            (2452e6, "CH9"), (2457e6, "CH10"), (2462e6, "CH11"), (2472e6, "CH13"),
        ],
        "markers_zoom_every": 1,
    },
]

GAPS = [
    (960e6,  1800e6, "Typically quiet — GSM uplink / other services"),
    (1880e6, 2110e6, "Guard band — usually low downlink activity"),
    (2170e6, 2400e6, "Guard band — usually low downlink activity"),
]

ISM_GRID = 1024
ISM_ROWS = 50

# ══════════════════════════════════════════════════════════════
#  ISM / Wi-Fi ANALYSIS (Component 1 report — 2400–2500 MHz)
# ══════════════════════════════════════════════════════════════
SAVE_MEASUREMENTS = True
OUTPUT_DIR = "measurements"

# Occupied if PSD (dBm/Hz) exceeds this level (adjust if needed)
OCCUPANCY_THRESHOLD_DB = -75.0

# Standard 2.4 GHz Wi-Fi channels for structure / occupancy tables
WIFI_CHANNELS = [
    {"ch": 1,  "mhz": 2412, "bw": 20},
    {"ch": 6,  "mhz": 2437, "bw": 20},
    {"ch": 11, "mhz": 2462, "bw": 20},
    {"ch": 2,  "mhz": 2417, "bw": 20},
    {"ch": 3,  "mhz": 2422, "bw": 20},
    {"ch": 4,  "mhz": 2427, "bw": 20},
    {"ch": 5,  "mhz": 2432, "bw": 20},
    {"ch": 7,  "mhz": 2442, "bw": 20},
    {"ch": 8,  "mhz": 2447, "bw": 20},
    {"ch": 9,  "mhz": 2452, "bw": 20},
    {"ch": 10, "mhz": 2457, "bw": 20},
    {"ch": 13, "mhz": 2472, "bw": 20},
]

LOG_EVERY_SWEEP = True          # append temporal row after each full sweep


def disp_range_mhz() -> str:
    return f"{DISP_START_MHZ:.0f}–{DISP_END_MHZ:.0f}"


def active_scan_regions():
    """Regions the Pluto sweeps — follows USE_BANDS_ONLY_SWEEP flag in code."""
    if USE_BANDS_ONLY_SWEEP:
        return list(SCAN_REGIONS_BANDS_ONLY)
    return list(SCAN_REGIONS)


def waterfall_freq_range():
    """
    Bottom plot frequency span.
    Wide view (includes ISM): zoom waterfall to 2.4 GHz for temporal detail.
    Narrow zoom: waterfall matches DISP_START–DISP_END.
    """
    span = DISP_END - DISP_START
    overlaps_ism = DISP_END >= ISM_BAND_LO and DISP_START <= ISM_BAND_HI
    if overlaps_ism and span > 500e6:
        lo = max(ISM_BAND_LO, DISP_START)
        hi = min(ISM_BAND_HI, DISP_END)
        return lo, hi
    return DISP_START, DISP_END


def regions_intersecting_display(regions):
    """Clip sweep regions to the configured display range."""
    out = []
    for rs, re in regions:
        lo = max(rs, DISP_START)
        hi = min(re, DISP_END)
        if hi > lo:
            out.append((lo, hi))
    return out or [(DISP_START, DISP_END)]


def band_markers_visible(band: dict) -> list[tuple[float, str]]:
    """More carrier lines when zoomed into a narrow band."""
    span_mhz = (DISP_END - DISP_START) / 1e6
    band_span = (band["f_end"] - band["f_start"]) / 1e6
    zoomed = span_mhz <= max(120.0, band_span * 2.5)

    if band["label"].startswith("GSM 900") and zoomed:
        every = band.get("markers_zoom_every", 2)
        return _gsm_carriers_mhz(935.2, 124, every=every)
    if band["label"].startswith("GSM 1800") and zoomed:
        every = band.get("markers_zoom_every", 5)
        return _gsm_carriers_mhz(1805.2, 374, every=every)
    return band.get("markers", [])


def is_ism_analysis_mode() -> bool:
    """True when display range is mainly the 2.4 GHz ISM band."""
    return DISP_START >= 2390e6 and DISP_END <= 2510e6


def band_label_at(freq_hz: float) -> str:
    for band in BANDS:
        if band["f_start"] <= freq_hz <= band["f_end"]:
            return band["label"]
    return "Spectrum"


def save_full_span_snapshot(freq_grid_hz: np.ndarray, psd_live: np.ndarray, psd_peak: np.ndarray):
    """Save PSD CSV + figure when display span is not ISM-only (e.g. 900–2500 MHz)."""
    if not SAVE_MEASUREMENTS:
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    freq_mhz = freq_grid_hz / 1e6

    psd_path = os.path.join(OUTPUT_DIR, f"psd_full_{tag}.csv")
    with open(psd_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["freq_mhz", "psd_live_dbm_hz", "psd_peak_dbm_hz"])
        for fm, live, peak in zip(freq_mhz, psd_live, psd_peak):
            w.writerow([f"{fm:.4f}", f"{live:.4f}", f"{peak:.4f}"])

    fig1, ax1 = plt.subplots(figsize=(12, 4), facecolor="white")
    ax1.plot(freq_mhz, psd_live, color="black", lw=0.6, label="Live PSD")
    ax1.plot(freq_mhz, psd_peak, color="orange", lw=0.6, alpha=0.7, label="Peak hold")
    ax1.axhline(-100, color="gray", ls="--", label="−100 dBm/Hz ref")
    for band in BANDS:
        if band["f_end"] < DISP_START or band["f_start"] > DISP_END:
            continue
        ax1.axvspan(
            band["f_start"] / 1e6, band["f_end"] / 1e6,
            color=band["shade"], alpha=0.15,
        )
        mid = (band["f_start"] + band["f_end"]) / 2 / 1e6
        ax1.text(mid, ax1.get_ylim()[1] * 0.85, band["label"], ha="center", fontsize=7)
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_ylabel("Power Spectral Density (dBm/Hz)")
    ax1.set_title(f"PSD full span — {disp_range_mhz()} MHz")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    fig1.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"figure_full_psd_{tag}.png")
    fig1.savefig(fig_path, dpi=150)
    plt.close(fig1)

    print(f"\n  Full-span snapshot saved in ./{OUTPUT_DIR}/")
    print(f"    psd_full_{tag}.csv")
    print(f"    figure_full_psd_{tag}.png")


def compute_live_metrics(freq_hz: np.ndarray, psd_live: np.ndarray) -> dict:
    """Quick stats for the live status bar (matches report analysis logic)."""
    freq_mhz = freq_hz / 1e6
    peak_idx = int(np.argmax(psd_live))
    noise_floor = float(np.percentile(psd_live, 25))
    threshold = max(OCCUPANCY_THRESHOLD_DB, noise_floor + 8.0)
    above = psd_live > threshold
    occ_bw_mhz = 0.0
    if np.any(above):
        idx = np.where(above)[0]
        occ_bw_mhz = float(freq_mhz[idx[-1]] - freq_mhz[idx[0]])
    return {
        "peak_db": float(psd_live[peak_idx]),
        "peak_mhz": float(freq_mhz[peak_idx]),
        "peak_hz": float(freq_hz[peak_idx]),
        "mean_db": float(np.mean(psd_live)),
        "noise_floor_db": noise_floor,
        "threshold_db": threshold,
        "occ_bw_mhz": occ_bw_mhz,
        "band": band_label_at(float(freq_hz[peak_idx])),
    }


class IsmAnalyzer:
    """PSD analysis for Wi-Fi channel structure, occupancy, and temporal logs."""

    def __init__(self, freq_grid_hz: np.ndarray):
        self.freq_mhz = freq_grid_hz / 1e6
        self.temporal_rows: list[dict] = []
        self.last_frame = -1
        if SAVE_MEASUREMENTS:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self.session_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.session_tag = ""

    def noise_floor_db(self, psd: np.ndarray) -> float:
        return float(np.percentile(psd, 25))

    def analyze(self, psd_live: np.ndarray, psd_peak: np.ndarray, frame_n: int) -> dict:
        nf = self.noise_floor_db(psd_live)
        threshold = max(OCCUPANCY_THRESHOLD_DB, nf + 8.0)
        channels = []

        for wc in WIFI_CHANNELS:
            lo = wc["mhz"] - wc["bw"] / 2
            hi = wc["mhz"] + wc["bw"] / 2
            mask = (self.freq_mhz >= lo) & (self.freq_mhz <= hi)
            if np.count_nonzero(mask) < 2:
                continue
            band_psd = psd_live[mask]
            band_peak = psd_peak[mask]
            peak_db = float(np.max(band_peak))
            occ_pct = 100.0 * float(np.mean(band_psd > threshold))
            occupied = occ_pct >= 25.0 or peak_db > threshold
            bw_est = self._estimate_occupancy_bw(self.freq_mhz[mask], band_psd, threshold)
            channels.append({
                "channel": wc["ch"],
                "centre_mhz": wc["mhz"],
                "peak_dbm_hz": round(peak_db, 2),
                "occupancy_pct": round(occ_pct, 1),
                "occupied": occupied,
                "est_bw_mhz": round(bw_est, 1),
                "status": "Busy" if occ_pct >= 50 else ("Light" if occupied else "Free"),
            })

        global_peak_idx = int(np.argmax(psd_peak))
        result = {
            "frame": frame_n,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "noise_floor_dbm_hz": round(nf, 2),
            "threshold_dbm_hz": round(threshold, 2),
            "global_peak_dbm_hz": round(float(psd_peak[global_peak_idx]), 2),
            "global_peak_mhz": round(float(self.freq_mhz[global_peak_idx]), 2),
            "channels": channels,
            "active_channels": sum(1 for c in channels if c["occupied"]),
        }
        return result

    @staticmethod
    def _estimate_occupancy_bw(freq_mhz, psd, threshold):
        above = psd > threshold
        if not np.any(above):
            return 0.0
        idx = np.where(above)[0]
        return float(freq_mhz[idx[-1]] - freq_mhz[idx[0]])

    def maybe_log(self, result: dict):
        if result["frame"] == self.last_frame:
            return
        self.last_frame = result["frame"]
        self.temporal_rows.append({
            "timestamp": result["timestamp"],
            "frame": result["frame"],
            "noise_floor_dbm_hz": result["noise_floor_dbm_hz"],
            "global_peak_mhz": result["global_peak_mhz"],
            "global_peak_dbm_hz": result["global_peak_dbm_hz"],
            "active_channels": result["active_channels"],
            **{f"ch{c['channel']}_occ_pct": c["occupancy_pct"] for c in result["channels"]},
            **{f"ch{c['channel']}_peak_db": c["peak_dbm_hz"] for c in result["channels"]},
        })
        self._print_table(result)

    def _print_table(self, result: dict):
        print(f"\n--- Wi-Fi ISM analysis | frame {result['frame']} | {result['timestamp']} ---")
        print(f"Noise floor: {result['noise_floor_dbm_hz']} dBm/Hz | "
              f"Threshold: {result['threshold_dbm_hz']} dBm/Hz | "
              f"Peak: {result['global_peak_dbm_hz']} dBm/Hz @ {result['global_peak_mhz']} MHz")
        print(f"{'Ch':>3} {'MHz':>6} {'Peak dBm/Hz':>12} {'Occupancy':>10} {'Est BW':>8} {'Status':>6}")
        for c in result["channels"]:
            if c["channel"] in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13):
                print(f"{c['channel']:>3} {c['centre_mhz']:>6} {c['peak_dbm_hz']:>12} "
                      f"{c['occupancy_pct']:>9.1f}% {c['est_bw_mhz']:>7.1f}M {c['status']:>6}")

    def save_all(self, freq_grid_hz, psd_live, psd_peak, wfall):
        if not SAVE_MEASUREMENTS:
            return
        tag = self.session_tag
        freq_mhz = freq_grid_hz / 1e6

        psd_path = os.path.join(OUTPUT_DIR, f"psd_{tag}.csv")
        with open(psd_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["freq_mhz", "psd_live_dbm_hz", "psd_peak_dbm_hz"])
            for fm, live, peak in zip(freq_mhz, psd_live, psd_peak):
                w.writerow([f"{fm:.4f}", f"{live:.4f}", f"{peak:.4f}"])

        if self.temporal_rows:
            tpath = os.path.join(OUTPUT_DIR, f"temporal_{tag}.csv")
            keys = list(self.temporal_rows[0].keys())
            with open(tpath, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.temporal_rows)

        self._save_report_figures(freq_mhz, psd_live, psd_peak, wfall, tag)
        print(f"\n  Measurements saved in ./{OUTPUT_DIR}/")
        print(f"    psd_{tag}.csv")
        if self.temporal_rows:
            print(f"    temporal_{tag}.csv")
        print(f"    figure1_psd_{tag}.png")
        print(f"    figure2_channels_{tag}.png")
        print(f"    figure3_occupancy_{tag}.png")
        if len(self.temporal_rows) > 1:
            print(f"    figure4_temporal_{tag}.png")

    def _save_report_figures(self, freq_mhz, psd_live, psd_peak, wfall, tag):
        # Figure 1 — PSD
        fig1, ax1 = plt.subplots(figsize=(10, 4), facecolor="white")
        ax1.plot(freq_mhz, psd_live, color="black", lw=0.8, label="Live PSD")
        ax1.plot(freq_mhz, psd_peak, color="orange", lw=0.8, alpha=0.7, label="Peak hold")
        ax1.axhline(-100, color="gray", ls="--", label="−100 dBm/Hz ref")
        for wc in WIFI_CHANNELS:
            if wc["ch"] in (1, 6, 11):
                ax1.axvline(wc["mhz"], color="red", alpha=0.25, ls=":")
                ax1.text(wc["mhz"], ax1.get_ylim()[1] * 0.9, f"Ch{wc['ch']}",
                         ha="center", fontsize=8, color="red")
        ax1.set_xlabel("Frequency (MHz)")
        ax1.set_ylabel("Power Spectral Density (dBm/Hz)")
        ax1.set_title(f"Figure 1 — PSD  {disp_range_mhz()} MHz")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        fig1.tight_layout()
        fig1.savefig(os.path.join(OUTPUT_DIR, f"figure1_psd_{tag}.png"), dpi=150)
        plt.close(fig1)

        last = self.analyze(psd_live, psd_peak, self.last_frame if self.last_frame >= 0 else 0)
        chs_all = last["channels"]
        chs_136 = [c for c in chs_all if c["channel"] in (1, 6, 11)]

        # Figure 2 — detected occupied channels (structure)
        if chs_all:
            fig2, ax2 = plt.subplots(figsize=(10, 3), facecolor="white")
            for c in chs_all:
                colour = "#E74C3C" if c["occupied"] else "#BDC3C7"
                ax2.barh(
                    f"Ch{c['channel']} ({c['centre_mhz']} MHz)",
                    c["peak_dbm_hz"],
                    color=colour,
                    alpha=0.85,
                )
            ax2.axvline(last["threshold_dbm_hz"], color="gray", ls="--",
                        label=f"Threshold {last['threshold_dbm_hz']} dBm/Hz")
            ax2.set_xlabel("Peak power (dBm/Hz)")
            ax2.set_title("Figure 2 — Detected Wi-Fi channels (red = occupied)")
            ax2.legend(fontsize=8)
            fig2.tight_layout()
            fig2.savefig(os.path.join(OUTPUT_DIR, f"figure2_channels_{tag}.png"), dpi=150)
            plt.close(fig2)

        if chs_136:
            fig3, ax3 = plt.subplots(figsize=(6, 4), facecolor="white")
            labels = [f"Ch{c['channel']}\n{c['centre_mhz']} MHz" for c in chs_136]
            occ = [c["occupancy_pct"] for c in chs_136]
            ax3.bar(labels, occ, color=["#4C9BE8", "#6BCB77", "#F5A623"][:len(chs_136)])
            ax3.set_ylabel("Occupancy (%)")
            ax3.set_title("Figure 3 — Channel occupancy (Ch 1 / 6 / 11)")
            ax3.set_ylim(0, 100)
            fig3.tight_layout()
            fig3.savefig(os.path.join(OUTPUT_DIR, f"figure3_occupancy_{tag}.png"), dpi=150)
            plt.close(fig3)

        if len(self.temporal_rows) > 1:
            fig4, ax4 = plt.subplots(figsize=(8, 4), facecolor="white")
            frames = [r["frame"] for r in self.temporal_rows]
            active = [r["active_channels"] for r in self.temporal_rows]
            ax4.plot(frames, active, "o-", color="purple")
            ax4.set_xlabel("Sweep frame #")
            ax4.set_ylabel("Active channels (count)")
            ax4.set_title("Figure 4 — Temporal variability (occupancy vs time)")
            ax4.grid(True, alpha=0.3)
            fig4.tight_layout()
            fig4.savefig(os.path.join(OUTPUT_DIR, f"figure4_temporal_{tag}.png"), dpi=150)
            plt.close(fig4)

def compute_psd_dbm(iq, fs, nfft, Z=50.0):
    n = min(len(iq), nfft)
    win = windows.hann(n)
    wp = np.sum(win ** 2)
    S = np.fft.fftshift(np.fft.fft(iq[:n] * win, n=nfft))
    psd = (np.abs(S) ** 2) / (wp * fs)
    dbm = 10 * np.log10(psd / Z * 1000 + 1e-20)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return f, dbm


def iter_sweep_chunks(sdr, f_start, f_end):
    """Yield (freq_hz, psd_dbm) slices as each 20 MHz chunk is captured."""
    step = SAMPLE_RATE * 0.85
    centers = []
    cf = f_start + SAMPLE_RATE / 2
    while cf < f_end - SAMPLE_RATE / 2:
        centers.append(cf)
        cf += step
    centers.append(f_end - SAMPLE_RATE / 2)

    for center in centers:
        f, avg = capture_psd(sdr, center)
        abs_f = f + center
        mask = (abs_f >= f_start) & (abs_f <= f_end)
        if np.count_nonzero(mask) < 2:
            continue
        yield abs_f[mask], avg[mask]


def sweep_region(sdr, f_start, f_end):
    """Sweep one contiguous region in ~20 MHz chunks. Returns (freqs, psd)."""
    af, ap = [], []
    for abs_f, avg in iter_sweep_chunks(sdr, f_start, f_end):
        af.append(abs_f)
        ap.append(avg)

    if not af:
        return np.array([]), np.array([])
    freqs = np.concatenate(af)
    psd = np.concatenate(ap)
    order = np.argsort(freqs)
    freqs = freqs[order]
    psd = psd[order]
    # Average duplicate frequency bins from overlapping chunks
    uniq, inv = np.unique(freqs, return_inverse=True)
    if len(uniq) < len(freqs):
        avg_psd = np.zeros_like(uniq, dtype=float)
        counts = np.zeros_like(uniq, dtype=float)
        np.add.at(avg_psd, inv, psd)
        np.add.at(counts, inv, 1)
        psd = avg_psd / counts
        freqs = uniq
    return freqs, psd


def connect_pluto(uri=PLUTO_URI, timeout=CONNECT_TIMEOUT):
  """Connect to Pluto without hanging indefinitely if USB is missing."""
  import adi

  result = {"sdr": None, "error": None}

  def _connect():
      try:
          result["sdr"] = adi.Pluto(uri=uri)
      except Exception as exc:
          result["error"] = exc

  thread = threading.Thread(target=_connect, daemon=True)
  thread.start()
  thread.join(timeout)
  if thread.is_alive():
      raise TimeoutError(
          f"No Pluto response within {timeout:.0f}s on '{uri}'. "
          "Check USB cable/driver, or run with --demo."
      )
  if result["error"] is not None:
      raise result["error"]
  if result["sdr"] is None:
      raise RuntimeError("Pluto connection failed for an unknown reason.")
  return result["sdr"]


def configure_rx(sdr, gain_db=RX_GAIN_DB):
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.rx_rf_bandwidth = int(SAMPLE_RATE)
    sdr.rx_buffer_size = FFT_SIZE * 4
    sdr.gain_control_mode_chan0 = RX_GAIN_MODE
    if RX_GAIN_MODE == "manual":
        sdr.rx_hardwaregain_chan0 = int(np.clip(gain_db, 0, 73))


def read_iq(sdr):
    raw = sdr.rx()
    iq = raw[0] if isinstance(raw, list) else raw
    return np.asarray(iq, dtype=np.complex64)


def capture_psd(sdr, center_hz):
    """Tune LO, flush buffer, then average PSD frames."""
    sdr.rx_lo = int(np.clip(center_hz, 325e6, 3800e6))
    time.sleep(LO_SETTLE_SEC)
    for _ in range(RX_FLUSH_FRAMES):
        read_iq(sdr)

    avg = None
    for _ in range(NUM_FRAMES):
        f, p = compute_psd_dbm(read_iq(sdr), SAMPLE_RATE, FFT_SIZE)
        avg = p if avg is None else avg + p
    return f, avg / NUM_FRAMES


def make_demo_psd(freq_grid, t):
    """Synthetic full-span spectrum for laptop testing without hardware."""
    raw = np.full(len(freq_grid), -102.0 + 0.4 * np.random.randn(len(freq_grid)))

    def add_bump(center_hz, width_hz, peak_db, n_peaks=1):
        nonlocal raw
        for k in range(n_peaks):
            c = center_hz + (k - (n_peaks - 1) / 2) * width_hz * 0.35
            raw += peak_db * np.exp(-0.5 * ((freq_grid - c) / width_hz) ** 2)

    add_bump(942e6, 0.12e6, 18, n_peaks=8)
    add_bump(1815e6, 0.15e6, 14, n_peaks=10)
    add_bump(2135e6, 4.5e6, 22, n_peaks=3)
    add_bump(2412e6 + 20e6 * np.sin(t * 0.7), 18e6, 28)
    add_bump(2442e6, 20e6, 20 + 4 * np.sin(t * 1.1))

    psd = np.full(len(freq_grid), -105.0)
    regions = active_scan_regions()
    if not USE_BANDS_ONLY_SWEEP and len(regions) == 1 and regions[0] == (DISP_START, DISP_END):
        psd[:] = raw[:]
    else:
        active = np.zeros(len(freq_grid), dtype=bool)
        for rs, re in regions_intersecting_display(regions):
            active |= (freq_grid >= rs) & (freq_grid <= re)
        psd[active] = raw[active]
    return np.clip(psd, -115, 5)


class SweepWorker:
    """Runs the slow SDR sweep off the GUI thread; UI reads snapshots."""

    def __init__(self, sdr=None, demo=False):
        self.sdr = sdr
        self.demo = demo
        self.freq_grid = np.linspace(DISP_START, DISP_END, GRID_PTS)
        self.wf_lo, self.wf_hi = waterfall_freq_range()
        self.ism_fg = np.linspace(self.wf_lo, self.wf_hi, ISM_GRID)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

        self.psd_live = np.full(GRID_PTS, -105.0)
        self.psd_peak = np.full(GRID_PTS, -200.0)
        self.wfall = np.full((ISM_ROWS, ISM_GRID), -100.0)
        self.frame_n = 0
        self.sweep_elapsed = 0.0
        self.status = "Initialising sweep..."

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def snapshot(self):
        with self._lock:
            return (
                self.psd_live.copy(),
                self.psd_peak.copy(),
                self.wfall.copy(),
                self.frame_n,
                self.sweep_elapsed,
                self.status,
            )

    def _apply_chunk(self, freqs, psd):
        if len(freqs) < 2:
            return
        lo, hi = float(freqs.min()), float(freqs.max())
        mask = (self.freq_grid >= lo) & (self.freq_grid <= hi)
        self.psd_live[mask] = np.interp(self.freq_grid[mask], freqs, psd)

    def _apply_region(self, rs, re, freqs, psd):
        mask = (self.freq_grid >= rs) & (self.freq_grid <= re)
        if len(freqs) < 2:
            return
        self.psd_live[mask] = np.interp(self.freq_grid[mask], freqs, psd)

    def ism_row_from_live(self, live):
        wf_mask = (self.freq_grid >= self.wf_lo) & (self.freq_grid <= self.wf_hi)
        if np.count_nonzero(wf_mask) < 2:
            return np.full(ISM_GRID, -105.0)
        return np.interp(self.ism_fg, self.freq_grid[wf_mask], live[wf_mask])

    def _run(self):
        t_demo = 0.0
        while not self._stop.is_set():
            t0 = time.time()
            try:
                if self.demo:
                    t_demo += 0.35
                    demo_psd = make_demo_psd(self.freq_grid, t_demo)
                    with self._lock:
                        self.psd_live[:] = demo_psd
                        self.psd_peak[:] = np.maximum(self.psd_peak, self.psd_live)
                        self.frame_n += 1
                        self.sweep_elapsed = time.time() - t0
                        self.status = f"Demo mode — live spectrum {disp_range_mhz()} MHz"
                    time.sleep(0.25)
                    continue

                regions = regions_intersecting_display(active_scan_regions())
                for rs, re in regions:
                    if self._stop.is_set():
                        break
                    chunk_n = 0
                    chunk_total = max(
                        1,
                        int(np.ceil((re - rs - SAMPLE_RATE) / (SAMPLE_RATE * 0.85))) + 1,
                    )
                    for abs_f, avg in iter_sweep_chunks(self.sdr, rs, re):
                        if self._stop.is_set():
                            break
                        chunk_n += 1
                        with self._lock:
                            self._apply_chunk(abs_f, avg)
                            self.psd_peak[:] = np.maximum(self.psd_peak, self.psd_live)
                            self.status = (
                                f"Sweeping {rs/1e6:.0f}–{re/1e6:.0f} MHz "
                                f"({chunk_n}/{chunk_total} chunks) @ "
                                f"{abs_f.mean()/1e6:.0f} MHz"
                            )

                with self._lock:
                    self.frame_n += 1
                    self.sweep_elapsed = time.time() - t0
                    self.status = (
                        f"Full sweep complete in {self.sweep_elapsed:.1f}s "
                        f"({disp_range_mhz()} MHz)"
                    )
            except Exception as exc:
                with self._lock:
                    self.status = f"Sweep error: {exc}"
                time.sleep(1.0)


def build_figure(worker):
    freq_grid = worker.freq_grid
    psd_live = worker.psd_live
    psd_peak = worker.psd_peak
    wfall = worker.wfall

    fig = plt.figure(figsize=(18, 9), facecolor="#080808")
    try:
        fig.canvas.manager.set_window_title("ENCS5323 Live Spectrum Sensor")
    except Exception:
        pass
    disp_mhz = disp_range_mhz()
    wf_lo_mhz = worker.wf_lo / 1e6
    wf_hi_mhz = worker.wf_hi / 1e6
    fig.suptitle(
        f"ENCS5323 – Live Spectrum Sensing  |  {disp_mhz} MHz  |  ADALM-PLUTO SDR",
        color="white", fontsize=13, fontweight="bold", y=0.99,
    )

    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.6, 1.0], hspace=0.40)
    ax_p = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1])

    for ax in (ax_p, ax_w):
        ax.set_facecolor("#080808")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        ax.xaxis.label.set_color("#bbbbbb")
        ax.yaxis.label.set_color("#bbbbbb")
        ax.title.set_color("white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#252525")

    ax_p.set_xlim(DISP_START / 1e6, DISP_END / 1e6)
    ax_p.set_ylim(-115, 10)
    ax_p.set_xlabel("Frequency (MHz)", fontsize=9)
    ax_p.set_ylabel("Power Spectral Density (dBm/Hz)", fontsize=9)
    ax_p.set_title(
        f"Live PSD  |  {disp_mhz} MHz  |  "
        "GSM 900 · GSM 1800 · UMTS 2100 · 2.4 GHz ISM",
        fontsize=9, pad=6,
    )
    ax_p.grid(True, color="#252525", lw=0.8)
    ax_p.axhline(-100, color="#ffcc66", lw=1.0, linestyle="--", label="−100 dBm/Hz noise ref", zorder=2)
    ref_occ = ax_p.axhline(
        OCCUPANCY_THRESHOLD_DB, color="#88cc44", lw=0.9, linestyle=":",
        label=f"Occ threshold {OCCUPANCY_THRESHOLD_DB:.0f} dBm/Hz", zorder=2,
    )

    for band in BANDS:
        if band["f_end"] < DISP_START or band["f_start"] > DISP_END:
            continue
        fs, fe = band["f_start"] / 1e6, band["f_end"] / 1e6
        ax_p.axvspan(fs, fe, color=band["shade"], alpha=band["shade_alpha"], zorder=0)
        ax_p.axvline(fs, color=band["shade"], lw=1.2, alpha=0.55, zorder=1)
        ax_p.axvline(fe, color=band["shade"], lw=1.2, alpha=0.55, zorder=1)
        mid = (fs + fe) / 2
        ax_p.text(
            mid, 8, band["label"], color=band["shade"], fontsize=8,
            ha="center", va="bottom", fontweight="bold", zorder=3,
            bbox=dict(facecolor="#111111", edgecolor=band["shade"], alpha=0.85, pad=2),
        )
        for freq, lbl in band_markers_visible(band):
            if not (DISP_START <= freq <= DISP_END):
                continue
            fm = freq / 1e6
            ax_p.axvline(fm, color=band["shade"], lw=0.55, alpha=0.45, linestyle=":", zorder=1)
            if lbl:
                ax_p.text(
                    fm, -112, lbl, color=band["shade"], fontsize=6,
                    ha="center", va="top", alpha=0.9, rotation=90, zorder=3,
                )

    for g_start, g_end, glabel in GAPS:
        if g_end < DISP_START or g_start > DISP_END:
            continue
        gmid = (g_start + g_end) / 2 / 1e6
        ax_p.text(gmid, -55, glabel, color="#333333", fontsize=6.5, ha="center", va="center", style="italic")

    line_live, = ax_p.plot(freq_grid / 1e6, psd_live, color="#e0e0e0", lw=0.8, label="Live PSD", zorder=4)
    line_peak, = ax_p.plot(
        freq_grid / 1e6, psd_peak, color="#ff6b35", lw=0.75, alpha=0.65,
        linestyle="--", label="Peak hold", zorder=4,
    )
    peak_marker = ax_p.axvline(
        DISP_START / 1e6, color="#00e5ff", lw=1.2, alpha=0.85, linestyle="-",
        label="Live peak", zorder=5,
    )
    ax_p.legend(loc="upper left", fontsize=8, facecolor="#111111", labelcolor="white", edgecolor="#333333")

    im_w = ax_w.imshow(
        wfall, aspect="auto", origin="upper",
        extent=[wf_lo_mhz, wf_hi_mhz, ISM_ROWS, 0],
        vmin=-100, vmax=0, cmap="inferno", interpolation="nearest",
    )
    ax_w.set_xlabel("Frequency (MHz)", fontsize=8)
    ax_w.set_ylabel("Time →", fontsize=7)
    ax_w.set_title(
        f"Waterfall – {wf_lo_mhz:.0f}–{wf_hi_mhz:.0f} MHz  |  Temporal Variability",
        fontsize=8, pad=4,
    )
    ax_w.set_xlim(wf_lo_mhz, wf_hi_mhz)
    ax_w.set_yticks([])

    for freq, lbl in BANDS[3].get("markers", []):
        if worker.wf_lo <= freq <= worker.wf_hi:
            ax_w.axvline(freq / 1e6, color="#ffffff", lw=0.6, alpha=0.35, linestyle="--")
            if lbl:
                ax_w.text(freq / 1e6, 1, lbl, color="#cccccc", fontsize=5, ha="center", va="bottom")

    cbar = fig.colorbar(im_w, ax=ax_w, orientation="vertical", pad=0.005, fraction=0.010)
    cbar.set_label("dBm/Hz", color="#aaaaaa", fontsize=7)
    cbar.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=6)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaaaaa")

    status = fig.text(0.01, 0.002, "Initialising live plot...", color="#aaaaaa", fontsize=8, va="bottom")
    return fig, ax_p, line_live, line_peak, im_w, status, peak_marker, ref_occ


def main():
    parser = argparse.ArgumentParser(description="ENCS5323 full-spectrum sensor")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run without Pluto hardware (synthetic 900–2500 MHz data)",
    )
    parser.add_argument("--uri", default=PLUTO_URI, help="Pluto URI (default: usb:)")
    parser.add_argument(
        "--gain", type=int, default=RX_GAIN_DB,
        help="RX gain in dB when using manual mode (0–73, default: %(default)s)",
    )
    args = parser.parse_args()

    print("=" * 62)
    print(f"  ENCS5323 – Live Spectrum Sensing  |  {disp_range_mhz()} MHz")
    print("  ADALM-PLUTO SDR  |  Component 1")
    print("=" * 62)
    if USE_BANDS_ONLY_SWEEP:
        print("  Sweep: bands-only mode (set USE_BANDS_ONLY_SWEEP = False in code for full span)")
    else:
        print(f"  Sweep: {disp_range_mhz()} MHz (matches DISP_START_MHZ / DISP_END_MHZ in code)")
    visible = [b["label"] for b in BANDS if b["f_end"] >= DISP_START and b["f_start"] <= DISP_END]
    print(f"  Bands on plot: {', '.join(visible) if visible else 'none in this MHz range'}")

    sdr = None
    if args.demo:
        print("\n  DEMO mode — no hardware required.\n")
    else:
        print(f"\n  Connecting ({args.uri}) ...")
        try:
            sdr = connect_pluto(args.uri)
            gain_db = int(np.clip(args.gain, 0, 73))
            configure_rx(sdr, gain_db=gain_db)
            print(f"  Connected. RX gain: {gain_db} dB ({RX_GAIN_MODE})")
            print(f"  Background sweep: {disp_range_mhz()} MHz...\n")
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            print("  Tip: use  python spectrum_sensor.py --demo  to test the UI.\n")
            return

    worker = SweepWorker(sdr=sdr, demo=args.demo)
    worker.start()

    ism_analyzer = IsmAnalyzer(worker.freq_grid) if is_ism_analysis_mode() else None
    if SAVE_MEASUREMENTS:
        if ism_analyzer:
            print(f"  ISM analysis ON — saving to ./{OUTPUT_DIR}/ on exit")
            print(f"  Occupancy threshold: {OCCUPANCY_THRESHOLD_DB} dBm/Hz (or noise floor + 8 dB)\n")
        else:
            print(f"  Full-span mode — PSD snapshot saves to ./{OUTPUT_DIR}/ on exit")
            print("  (no Wi-Fi channel tables; use 2400–2500 MHz for Figures 1–4)\n")

    fig, ax_p, line_live, line_peak, im_w, status, peak_marker, ref_occ = build_figure(worker)
    wfall_display = worker.wfall.copy()
    last_ui_tick = [time.time()]
    last_logged_frame = [-1]
    session_t0 = time.time()
    fps_times: list[float] = []

    def update(_):
        live, peak, wfall, frame_n, elapsed, stat = worker.snapshot()
        line_live.set_ydata(live)
        line_peak.set_ydata(peak)

        m = compute_live_metrics(worker.freq_grid, live)
        peak_marker.set_xdata([m["peak_mhz"], m["peak_mhz"]])
        ref_occ.set_ydata([m["threshold_db"], m["threshold_db"]])

        ch1_hint = ""
        if ism_analyzer:
            analysis = ism_analyzer.analyze(live, peak, frame_n)
            if LOG_EVERY_SWEEP and frame_n != last_logged_frame[0]:
                ism_analyzer.maybe_log(analysis)
                last_logged_frame[0] = frame_n
            ch1 = next((c for c in analysis["channels"] if c["channel"] == 1), None)
            if ch1:
                ch1_hint = f"  |  Ch1 occ {ch1['occupancy_pct']:.0f}%"

        # Roll waterfall every UI frame (~5 Hz) so it fills in seconds, not minutes.
        now = time.time()
        fps_times.append(now)
        if len(fps_times) > 30:
            fps_times.pop(0)
        ui_fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 1e-3) if len(fps_times) > 1 else 0.0
        session_sec = now - session_t0

        if now - last_ui_tick[0] >= 0.2:
            ism_row = worker.ism_row_from_live(live)
            wfall_display[:] = np.roll(wfall_display, -1, axis=0)
            wfall_display[-1, :] = ism_row
            last_ui_tick[0] = now

        im_w.set_data(wfall_display)
        wf_lo = float(np.percentile(wfall_display, 10))
        wf_hi = float(np.percentile(wfall_display, 99.5) + 3)
        im_w.set_clim(max(-110, wf_lo), min(0, max(wf_hi, wf_lo + 15)))

        status.set_text(
            f"{m['band']}  |  "
            f"Peak: {m['peak_db']:.1f} dBm/Hz @ {m['peak_mhz']:.3f} MHz  |  "
            f"Mean: {m['mean_db']:.1f} dBm/Hz  |  "
            f"Noise: {m['noise_floor_db']:.1f} dBm/Hz  |  "
            f"Occ BW: {m['occ_bw_mhz']:.1f} MHz"
            f"{ch1_hint}  |  "
            f"{ui_fps:.1f} fps  |  t={session_sec:.0f}s  |  frame #{frame_n}"
        )
        return line_live, line_peak, im_w, status, peak_marker, ref_occ

    # Must keep a reference — otherwise matplotlib garbage-collects the animation
    # and update() never runs (plot stays flat at the -105 dBm default).
    ani = animation.FuncAnimation(
        fig, update, interval=UPDATE_INTERVAL, blit=False, cache_frame_data=False,
    )
    fig._spectrum_ani = ani

    def on_close(_):
        worker.stop()
        if SAVE_MEASUREMENTS:
            live, peak, wfall, _, _, _ = worker.snapshot()
            if ism_analyzer:
                ism_analyzer.save_all(worker.freq_grid, live, peak, wfall)
            else:
                save_full_span_snapshot(worker.freq_grid, live, peak)
        if sdr is not None:
            try:
                sdr.rx_destroy_buffer()
            except Exception:
                pass

    fig.canvas.mpl_connect("close_event", on_close)
    plt.ion()
    plt.tight_layout(rect=[0, 0.015, 1, 0.97])
    plt.show(block=True)
    worker.stop()
    print("  Stopped.")


if __name__ == "__main__":
    main()
