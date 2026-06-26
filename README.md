# Real-World Spectrum Sensing and Controlled RF Analysis

Software Defined Radio project using the ADALM-PLUTO SDR platform.  
Two independent, concurrently-operable components: a live spectrum sensor (Component 1) and a controlled 2.4 GHz interference generator (Component 2).

---

## Hardware Requirements

- Two ADALM-PLUTO SDR units connected via USB
  - **Unit 1 (RX)** – spectrum sensor
  - **Unit 2 (TX)** – interference generator
- A Wi-Fi access point or client device (your own equipment) for interference measurements

---

## Installation

```bash
pip install pyadi-iio numpy matplotlib scipy
```

Verify that `libiio` and the Pluto USB drivers are installed on your system. See the [ADALM-PLUTO documentation](https://wiki.analog.com/university/tools/pluto) for platform-specific setup.

---

## Component 1 — Spectrum Sensor (`spectrum_sensor.py`)

Continuously sweeps a configurable frequency range, displaying a live power spectral density plot and a rolling waterfall diagram. When pointed at the 2.4 GHz ISM band, it also logs per-channel occupancy statistics to the `measurements/` directory on exit.

### Configuration

Open `spectrum_sensor.py` and edit the two variables near the top of the file:

```python
DISP_START_MHZ = 2400   # start of display and sweep range
DISP_END_MHZ   = 2500   # end of display and sweep range
```

Common presets (uncomment one pair):

| Band | Start MHz | End MHz |
|---|---|---|
| GSM 900 downlink | 935 | 960 |
| GSM 1800 downlink | 1805 | 1880 |
| UMTS 2100 downlink | 2110 | 2170 |
| 2.4 GHz ISM (Wi-Fi / BT) | 2400 | 2500 |

Other tunable parameters:

| Variable | Default | Description |
|---|---|---|
| `RX_GAIN_DB` | 50 | RX gain in dB (0–73). Raise if the plot is flat; lower if clipping. |
| `RX_GAIN_MODE` | `"manual"` | `"manual"` or `"slow_attack"` |
| `NUM_FRAMES` | 2 | FFT frames averaged per 20 MHz chunk (lower = faster sweep) |
| `OCCUPANCY_THRESHOLD_DB` | -75.0 | dBm/Hz level above which a channel is considered occupied |
| `SAVE_MEASUREMENTS` | `True` | Save PSD snapshots and occupancy CSV to `measurements/` on exit |

### Running

```bash
# With hardware (RX Pluto on usb:)
python spectrum_sensor.py

# Different URI or gain
python spectrum_sensor.py --uri usb:1 --gain 60

# Demo mode — no hardware needed, synthetic data
python spectrum_sensor.py --demo
```

### Output

Closing the plot window saves the following to `measurements/`:

- `psd_snapshot_<timestamp>.csv` — live and peak PSD across the sweep
- `occupancy_<timestamp>.csv` — per-channel occupancy percentages (ISM mode only)
- `temporal_log_<timestamp>.csv` — per-sweep occupancy rows (ISM mode only)

---

## Component 2 — Interference Generator (`interference_generator.py`)

Transmits a band-limited interference signal within the 2.4 GHz ISM band and optionally monitors a ping target during transmission to measure latency/packet-loss degradation.

> **Safety notice:** Operate indoors only, at minimum necessary power, directed exclusively at your own equipment. Transmission must not affect any equipment or persons outside your experiment. Default duration is 45 seconds; maximum is 120 seconds.

### Configuration

Open `interference_generator.py` and set the TX parameters:

```python
TX_CENTER_MHZ    = 2412   # centre frequency of the interference signal
TX_BANDWIDTH_MHZ = 20     # spectral width of the interference (MHz)
USE_WIFI_CHANNEL = None   # set to 1–13 to override TX_CENTER_MHZ with a standard Wi-Fi channel
```

Wi-Fi channel centre frequencies for reference:

| Channel | Centre (MHz) | Typical 20 MHz range |
|---|---|---|
| 1 | 2412 | 2402–2422 |
| 6 | 2437 | 2427–2447 |
| 11 | 2462 | 2452–2472 |

The interference must remain inside 2400–2483.5 MHz. The script validates this and will error before transmitting if the configured range exceeds the ISM band.

### Running

```bash
# Default: noise, 45 s, -30 dB attenuation, centre/BW from code
python interference_generator.py

# With ping monitoring (replace 192.168.1.1 with your gateway IP)
python interference_generator.py --ping 192.168.1.1

# Single-tone (CW) waveform, 60 seconds, slightly stronger signal
python interference_generator.py --waveform cw --duration 60 --gain -20

# Demo mode — no hardware, no RF transmitted
python interference_generator.py --demo
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--uri` | `usb:` | Pluto URI. Use `usb:1` if the TX unit is second. |
| `--gain` | -30.0 | TX attenuation in dB. Range: -80 to -10. More negative = weaker. |
| `--duration` | 45 | Transmit time in seconds (5–120). Use `0` to run until Ctrl+C. |
| `--waveform` | `noise` | `noise` (AWGN) or `cw` (single tone). |
| `--ping HOST` | — | Ping HOST once per second during TX and print a loss/RTT summary. |
| `--demo` | — | Simulate TX without transmitting any RF. |

---

## Running Both Components Simultaneously

Start each script in a separate terminal. Component 1 will display the spectral impact of the interference signal generated by Component 2 in real time.

```bash
# Terminal 1 — set DISP_START_MHZ = 2400, DISP_END_MHZ = 2500 in code first
python spectrum_sensor.py

# Terminal 2 — set TX_CENTER_MHZ and TX_BANDWIDTH_MHZ in code first
python interference_generator.py --ping 192.168.1.1
```

If both Pluto units share the same USB hub, specify URIs explicitly:

```bash
python spectrum_sensor.py --uri usb:0
python interference_generator.py --uri usb:1
```

---

## Suggested Experiments

1. **Full-channel vs half-channel interference** — set `TX_BANDWIDTH_MHZ = 20` then `10` on the same Wi-Fi channel; compare ping loss and throughput degradation.
2. **Channel offset** — centre interference on Ch 6 while the AP operates on Ch 1; observe the partial overlap in the waterfall.
3. **Waveform comparison** — repeat any experiment with `--waveform cw` vs `--waveform noise` and compare spectral footprint and receiver impact.
4. **Power sweep** — repeat at `--gain -30`, `-20`, `-15` dB; record the threshold at which measurable packet loss begins.

---

## Troubleshooting

**No Pluto detected**

```bash
iio_info -s          # list all IIO devices
```

Ensure `libiio` is installed and the Pluto USB driver is loaded. Try unplugging and replugging. If two Plutos are connected, use `--uri usb:0` and `--uri usb:1` to address them separately.

**Plot is flat / all noise floor**

- Increase `RX_GAIN_DB` (try 60–70 for GSM bands, 50 for ISM).
- Check antenna is attached to the RX Pluto.
- Confirm `DISP_START_MHZ` / `DISP_END_MHZ` cover a band with active signals.

**TX error on interference generator**

- Confirm the TX Pluto is not also being used as the RX unit.
- Check that `TX_CENTER_MHZ ± TX_BANDWIDTH_MHZ/2` falls within 2400–2483.5 MHz.
- Try `--demo` first to verify the script runs without hardware.
