"""
ENCS5323 Project – Component 2: Controlled 2.4 GHz ISM Interference Generator
Device : ADALM-PLUTO SDR (TX unit — use the SECOND Pluto, not the RX one)

Generates a band-limited interference signal in the 2.4 GHz ISM band and optionally
records Wi-Fi receiver metrics (ping latency / packet loss) while transmitting.

SAFETY (required by the assignment):
  - Indoor use only, minimum duration (default 45 s per run)
  - Low TX power — default attenuation -30 dB (raise only if needed)
  - Aim only at YOUR group's Wi-Fi equipment in the same room
  - Stop immediately if anyone outside your group is affected

Install:  pip install pyadi-iio numpy
Run:      python interference_generator.py
          (edit TX_CENTER_MHZ / TX_BANDWIDTH_MHZ in code first)
Demo:     python interference_generator.py --demo
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import threading
import time

import numpy as np

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

PLUTO_URI = "usb:"
SAMPLE_RATE = 20e6
BUFFER_SIZE = 2**18
CONNECT_TIMEOUT = 8.0

# TX gain on Pluto is attenuation: 0 dB = maximum, more negative = weaker.
# Keep weak for indoor academic use (-30 dB is a safe starting point).
DEFAULT_TX_GAIN_DB = -30.0
MIN_TX_GAIN_DB = -80.0
MAX_TX_GAIN_DB = -10.0          # hard safety cap (do not go to 0 indoors)

DEFAULT_DURATION_SEC = 45.0
MAX_DURATION_SEC = 120.0

ISM_MIN_HZ = 2400e6
ISM_MAX_HZ = 2483.5e6           # must stay inside 2.4 GHz ISM (assignment + safety)

# ══════════════════════════════════════════════════════════════
#  TX FREQUENCY — edit here only (no command-line frequency args)
# ══════════════════════════════════════════════════════════════
#
#  Centre = middle of the interference (MHz)
#  Bandwidth = width of the interference (MHz)
#  Occupied range ≈ [centre - BW/2 , centre + BW/2]
#
#  Examples:
#    Wi-Fi Ch1, full 20 MHz channel:  centre 2412, BW 20  → ~2402–2422 MHz
#    Wi-Fi Ch6, half channel:         centre 2437, BW 10  → ~2432–2442 MHz
#    Wi-Fi Ch11:                      centre 2462, BW 20
#
TX_CENTER_MHZ = 2412
TX_BANDWIDTH_MHZ = 20

# Optional: set a Wi-Fi channel number instead of TX_CENTER_MHZ (1–13, or None)
USE_WIFI_CHANNEL = None   # e.g. 6 → 2437 MHz; overrides TX_CENTER_MHZ when set

# Wi-Fi channel → centre frequency (MHz) lookup
WIFI_CHANNELS_MHZ = {
    1: 2412, 2: 2417, 3: 2422, 4: 2427, 5: 2432, 6: 2437,
    7: 2442, 8: 2447, 9: 2452, 10: 2457, 11: 2462, 12: 2467, 13: 2472,
}


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def connect_pluto(uri: str, timeout: float = CONNECT_TIMEOUT):
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
            f"No Pluto TX response within {timeout:.0f}s on '{uri}'. "
            "Connect the second Pluto or check USB order (usb: / usb:1)."
        )
    if result["error"] is not None:
        raise result["error"]
    if result["sdr"] is None:
        raise RuntimeError("Pluto TX connection failed.")
    return result["sdr"]


def clamp_tx_gain(gain_db: float) -> float:
    return float(np.clip(gain_db, MIN_TX_GAIN_DB, MAX_TX_GAIN_DB))


def validate_ism(center_hz: float, bandwidth_hz: float) -> None:
    if bandwidth_hz <= 0 or bandwidth_hz > SAMPLE_RATE:
        raise ValueError(f"Bandwidth must be in (0, {SAMPLE_RATE/1e6:.0f}] MHz")
    half = bandwidth_hz / 2
    if center_hz - half < ISM_MIN_HZ or center_hz + half > ISM_MAX_HZ + 16e6:
        raise ValueError(
            f"Signal must stay inside 2.4 GHz ISM band "
            f"({ISM_MIN_HZ/1e6:.0f}–{ISM_MAX_HZ/1e6:.0f} MHz). "
            f"Got centre {center_hz/1e6:.1f} MHz, BW {bandwidth_hz/1e6:.1f} MHz."
        )


def band_limited_waveform(
    n: int,
    fs: float,
    bandwidth_hz: float,
    waveform: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one cyclic buffer of complex baseband samples."""
    bw = min(bandwidth_hz, fs * 0.95)
    t = np.arange(n) / fs

    if waveform == "cw":
        # Single-tone at centre of passband (baseband DC after filtering)
        x = np.ones(n, dtype=np.complex64)
    elif waveform == "noise":
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    else:
        raise ValueError(f"Unknown waveform: {waveform}")

    # Frequency-domain brick-wall band-limit
    X = np.fft.fftshift(np.fft.fft(x))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / fs))
    mask = np.abs(freqs) <= bw / 2
    X[~mask] = 0
    y = np.fft.ifft(np.fft.ifftshift(X)).astype(np.complex64)

    peak = np.max(np.abs(y)) or 1.0
    y /= peak
    y *= 2**14
    return y


def configure_tx(sdr, center_hz: float, bandwidth_hz: float, gain_db: float) -> None:
    sdr.sample_rate = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth = int(min(SAMPLE_RATE, max(bandwidth_hz, 1e6)))
    sdr.tx_lo = int(center_hz)
    sdr.tx_hardwaregain_chan0 = gain_db
    sdr.tx_cyclic_buffer = True


def ping_once(host: str, timeout_ms: int = 1000) -> tuple[float | None, bool]:
    """Return (RTT ms, success)."""
    system = platform.system().lower()
    if system == "windows":
        cmd = ["ping", "-n", "1", "-w", str(timeout_ms), host]
    else:
        cmd = ["ping", "-c", "1", "-W", str(max(1, timeout_ms // 1000)), host]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=(timeout_ms / 1000) + 2,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None, False

    if proc.returncode != 0:
        return None, False

    out = proc.stdout.replace(",", ".")
    for token in out.split():
        if token.endswith("ms") and token[:-2].replace(".", "", 1).isdigit():
            return float(token[:-2]), True
        if "time=" in token.lower():
            try:
                return float(token.lower().split("time=")[-1].rstrip("ms")), True
            except ValueError:
                pass
        if "time<" in token.lower():
            return 0.5, True
    return None, True


class PingMonitor:
    def __init__(self, host: str, interval_sec: float = 1.0):
        self.host = host
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[tuple[float, float | None, bool]] = []

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self):
        while not self._stop.is_set():
            t0 = time.time()
            rtt, ok = ping_once(self.host)
            self.samples.append((time.time(), rtt, ok))
            time.sleep(max(0.0, self.interval_sec - (time.time() - t0)))

    def summary(self) -> str:
        if not self.samples:
            return "no ping samples"
        oks = [s for s in self.samples if s[2]]
        rtts = [s[1] for s in oks if s[1] is not None]
        loss = 100.0 * (1 - len(oks) / len(self.samples))
        if rtts:
            return (
                f"ping {self.host}: n={len(self.samples)}, loss={loss:.0f}%, "
                f"RTT avg={np.mean(rtts):.1f} ms, max={np.max(rtts):.1f} ms"
            )
        return f"ping {self.host}: n={len(self.samples)}, loss={loss:.0f}%"


def print_safety_banner():
    print("!" * 62)
    print("  SAFETY: Indoor - low power - your equipment only - max 60 s")
    print("  Stop TX if anyone outside your group may be affected.")
    print("!" * 62)


def run_transmit(
    sdr,
    center_hz: float,
    bandwidth_hz: float,
    gain_db: float,
    duration_sec: float,
    waveform: str,
    ping_host: str | None,
    demo: bool,
) -> None:
    validate_ism(center_hz, bandwidth_hz)
    gain_db = clamp_tx_gain(gain_db)
    if duration_sec > 0:
        duration_sec = min(max(5.0, duration_sec), MAX_DURATION_SEC)

    print(f"\n  Centre frequency : {center_hz/1e6:.2f} MHz")
    print(f"  Bandwidth        : {bandwidth_hz/1e6:.1f} MHz")
    print(f"  TX gain          : {gain_db:.1f} dB (0 = max power)")
    print(f"  Waveform         : {waveform}")
    if duration_sec <= 0:
        print(f"  Duration         : until Ctrl+C")
    else:
        print(f"  Duration         : {duration_sec:.0f} s")
    if ping_host:
        ping_host = ping_host.strip().rstrip("\\/")
        print(f"  Ping target      : {ping_host}")

    rng = np.random.default_rng(42)
    iq = band_limited_waveform(BUFFER_SIZE, SAMPLE_RATE, bandwidth_hz, waveform, rng)

    monitor = None
    if ping_host:
        monitor = PingMonitor(ping_host, interval_sec=1.0)
        monitor.start()

    if demo:
        print("\n  DEMO — not transmitting RF.")
        for sec in range(int(duration_sec)):
            print(f"  ... simulating TX {sec+1}/{int(duration_sec)} s", end="\r")
            time.sleep(1.0)
        print()
    else:
        configure_tx(sdr, center_hz, bandwidth_hz, gain_db)
        sdr.tx(iq)
        print("\n  TX ON — press Ctrl+C to stop early")
        print(f"  (auto-stops after {duration_sec:.0f} s — use --duration 0 to run until Ctrl+C)\n")
        t_tx0 = time.time()
        try:
            if duration_sec <= 0:
                while True:
                    time.sleep(1.0)
            else:
                time.sleep(duration_sec)
        except KeyboardInterrupt:
            print("\n  Stopped by user.")
        finally:
            elapsed = time.time() - t_tx0
            try:
                sdr.tx_destroy_buffer()
            except Exception:
                pass
            print(f"  TX OFF.  (transmitted {elapsed:.1f} s)")

    if monitor is not None:
        monitor.stop()
        print(f"  {monitor.summary()}")


def get_tx_center_and_bandwidth() -> tuple[float, float]:
    """Read centre + bandwidth from code configuration (lines ~65–66)."""
    if USE_WIFI_CHANNEL is not None:
        if USE_WIFI_CHANNEL not in WIFI_CHANNELS_MHZ:
            raise ValueError(f"USE_WIFI_CHANNEL must be 1–13, got {USE_WIFI_CHANNEL}")
        center_mhz = WIFI_CHANNELS_MHZ[USE_WIFI_CHANNEL]
    else:
        center_mhz = TX_CENTER_MHZ
    return center_mhz * 1e6, TX_BANDWIDTH_MHZ * 1e6


def tx_range_mhz() -> str:
    center_mhz = (
        WIFI_CHANNELS_MHZ[USE_WIFI_CHANNEL]
        if USE_WIFI_CHANNEL is not None
        else TX_CENTER_MHZ
    )
    lo = center_mhz - TX_BANDWIDTH_MHZ / 2
    hi = center_mhz + TX_BANDWIDTH_MHZ / 2
    return f"{lo:.1f}–{hi:.1f} MHz (centre {center_mhz:.0f} MHz, BW {TX_BANDWIDTH_MHZ:.0f} MHz)"


def main():
    parser = argparse.ArgumentParser(
        description="ENCS5323 controlled 2.4 GHz ISM interference generator (Pluto TX)",
    )
    parser.add_argument("--uri", default=PLUTO_URI, help="Pluto URI (default: usb:)")
    parser.add_argument("--demo", action="store_true", help="Simulate without transmitting")
    parser.add_argument(
        "--gain", "--atten", type=float, default=DEFAULT_TX_GAIN_DB, dest="gain",
        help=f"TX attenuation dB ({MIN_TX_GAIN_DB}..{MAX_TX_GAIN_DB}, default: %(default)s)",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_SEC,
        help=f"Transmit seconds (default: %(default)s). Use 0 to run until Ctrl+C.",
    )
    parser.add_argument(
        "--waveform", "--type", dest="waveform", default="noise",
        choices=["noise", "cw", "wideband", "narrowband", "tone", "sweep"],
        help="noise/wideband=AWGN, cw/tone=single tone (sweep/narrowband -> noise)",
    )
    parser.add_argument(
        "--ping", metavar="HOST",
        help="Ping HOST every second during TX (e.g. gateway IP)",
    )
    args = parser.parse_args()

    # Map alternate --type names to waveforms
    type_map = {
        "wideband": "noise", "narrowband": "noise",
        "tone": "cw", "sweep": "noise",
    }
    args.waveform = type_map.get(args.waveform, args.waveform)

    print("=" * 62)
    print("  ENCS5323 – Controlled Interference Generator")
    print("  2.4 GHz ISM  |  ADALM-PLUTO TX")
    print("=" * 62)
    print_safety_banner()
    print(f"  TX from code: {tx_range_mhz()}\n")

    try:
        center_hz, bandwidth_hz = get_tx_center_and_bandwidth()
    except ValueError as exc:
        print(f"\n  ERROR: {exc}\n")
        return

    sdr = None
    if not args.demo:
        print(f"\n  Connecting TX Pluto ({args.uri}) ...")
        try:
            sdr = connect_pluto(args.uri)
            print("  Connected.\n")
        except Exception as exc:
            print(f"\n  ERROR: {exc}")
            print("  Tip: list devices with  iio_info -s")
            print("  Or test UI timing:  python interference_generator.py --demo\n")
            return

    try:
        run_transmit(
            sdr=sdr,
            center_hz=center_hz,
            bandwidth_hz=bandwidth_hz,
            gain_db=args.gain,
            duration_sec=args.duration,
            waveform=args.waveform,
            ping_host=args.ping,
            demo=args.demo,
        )
    except ValueError as exc:
        print(f"\n  ERROR: {exc}\n")
    except Exception as exc:
        print(f"\n  TX error: {exc}\n")

    print("\n  Done.")
    print("  Change TX in code: TX_CENTER_MHZ, TX_BANDWIDTH_MHZ (or USE_WIFI_CHANNEL)\n")


if __name__ == "__main__":
    main()
