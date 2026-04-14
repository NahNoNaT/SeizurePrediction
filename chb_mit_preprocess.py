import re
import glob
import json
import argparse
import numpy as np
from scipy.signal import stft as scipy_stft
import warnings
import time

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# PAPER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
FS            = 256              # Hz — Sec II.A
N_CH          = 22               # bipolar 10-20 — Sec II.A
PREICTAL_SEC  = 30 * 60          # 30 min — Sec II.A
INTERV_SEC    = 5 * 60           # 5 min intervention — Sec II.A
GAP_SEC       = 4 * 3600         # 4h interictal gap — Sec II.A
LEAD_GAP_SEC  = 30 * 60          # 30 min leading criterion — Sec II.A
SEG_SEC       = 5                # default window — Sec II.B

# STFT → (9 time, 114 freq) per 5s segment — Table II
NPERSEG  = 64
NOVERLAP = 32 
NFFT     = 256

# Noise bins to remove — Sec II.B
DC_BINS       = {0}                       # 0 Hz
POWERLINE_1   = set(range(57, 64))        # 57-63 Hz
POWERLINE_2   = set(range(117, 124))      # 117-123 Hz
REMOVE_BINS   = DC_BINS | POWERLINE_1 | POWERLINE_2   # 15 total
KEEP_INDICES  = sorted(set(range(NFFT//2+1)) - REMOVE_BINS)  # 114 bins
assert len(KEEP_INDICES) == 114

# Standard 22-channel bipolar montage
CH_22 = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ', 'CZ-PZ',
    'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8',
]


# ═══════════════════════════════════════════════════════════════════════
# 1. PARSE SUMMARY FILE
# ═══════════════════════════════════════════════════════════════════════

def parse_summary(patient_dir):
    """
    Parse chbXX-summary.txt → {filename: [(start_sec, end_sec), ...]}
    Also returns file_order: list of filenames in CHRONOLOGICAL order
    (summary files list them chronologically, unlike glob which is alphabetical)
    """
    sfiles = glob.glob(os.path.join(patient_dir, "*summary*"))
    if not sfiles:
        raise FileNotFoundError(f"No summary file in {patient_dir}")

    path = sfiles[0]
    result = {}
    file_order = []  # Chronological order from summary
    cur = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("File Name:"):
            cur = line.split(":", 1)[1].strip()
            result.setdefault(cur, [])
            file_order.append(cur)

        elif re.match(r"Seizure\s*\d*\s*Start\s*Time", line, re.IGNORECASE) and cur:
            nums = re.findall(r"(\d+)", line.split("Time")[-1])
            if nums:
                start = int(nums[0])
                i += 1
                if i < len(lines):
                    eline = lines[i].strip()
                    enums = re.findall(r"(\d+)", eline.split("Time")[-1])
                    if enums:
                        end = int(enums[0])
                        result[cur].append((start, end))
        i += 1

    sz_files = {k: v for k, v in result.items() if v}
    total_sz = sum(len(v) for v in sz_files.values())
    print(f"  Summary: {os.path.basename(path)}")
    print(f"  Files: {len(result)} | With seizures: {len(sz_files)} | Seizures: {total_sz}")
    for fname, szs in sorted(sz_files.items()):
        for s, e in szs:
            print(f"    {fname}: {s}s → {e}s  (dur={e-s}s)")

    return result, file_order


# ═══════════════════════════════════════════════════════════════════════
# 2. READ EDF + 22-CHANNEL SELECTION
# ═══════════════════════════════════════════════════════════════════════

def read_edf(filepath):
    """Read .edf → (data[n_ch, n_samp], ch_names, fs)"""
    try:
        import pyedflib
        f = pyedflib.EdfReader(filepath)
        n_ch = f.signals_in_file
        ch_names = f.getSignalLabels()
        fs = int(f.getSampleFrequency(0))
        n_samps = f.getNSamples()
        min_samp = min(n_samps)
        data = np.zeros((n_ch, min_samp))
        for c in range(n_ch):
            sig = f.readSignal(c)
            data[c] = sig[:min_samp]
        f.close()
        return data, ch_names, fs
    except ImportError:
        pass

    try:
        import mne
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        return raw.get_data(), raw.ch_names, int(raw.info["sfreq"])
    except ImportError:
        pass

    raise ImportError("Cần cài: pip install pyedflib  HOẶC  pip install mne")


def _norm_ch(name):
    """Normalize channel name for matching."""
    s = name.strip().upper().replace(" ", "").replace(".", "-")
    for old, new in [("T3", "T7"), ("T4", "T8"), ("T5", "P7"), ("T6", "P8")]:
        s = s.replace(old, new)
    return s


def select_22ch(data, ch_names):
    """
    Map to standard 22 channels.
    Handles: naming variations, extra channels, blank '-' channels (CHB14),
    duplicate channel names (CHB14 has T8-P8 twice).
    """
    # Build lookup, skip blank channels named '-' or ''
    lookup = {}
    for i, name in enumerate(ch_names):
        n = _norm_ch(name)
        if n in ("", "-"):
            continue  # Skip CHB14's blank channels
        if n not in lookup:  # Keep first occurrence of duplicates
            lookup[n] = i

    n_samp = data.shape[1]
    out = np.zeros((22, n_samp), dtype=np.float64)
    matched = 0

    for j, std in enumerate(CH_22):
        key = _norm_ch(std)
        if key in lookup:
            out[j] = data[lookup[key]]
            matched += 1
            continue
        parts = key.split("-")
        if len(parts) == 2:
            rev = parts[1] + "-" + parts[0]
            if rev in lookup:
                out[j] = -data[lookup[rev]]
                matched += 1
                continue
        for cname, cidx in lookup.items():
            if key.replace("-", "") in cname.replace("-", ""):
                out[j] = data[cidx]
                matched += 1
                break

    return out, matched


# ═══════════════════════════════════════════════════════════════════════
# 3. STFT SPECTROGRAM — (22, 9, 114)
# ═══════════════════════════════════════════════════════════════════════

def stft_1ch(sig):
    """1D signal → (n_time, 114). For 5s: (9, 114)."""
    _, _, Zxx = scipy_stft(sig, fs=FS, nperseg=NPERSEG,
                           noverlap=NOVERLAP, nfft=NFFT,
                           boundary=None, padded=False)
    log_mag = np.log1p(np.abs(Zxx))
    return log_mag.T[:, KEEP_INDICES]


def stft_22ch(seg22):
    """(22, n_samples) → (22, n_time, 114)"""
    return np.stack([stft_1ch(seg22[c]) for c in range(22)], axis=0)


# ═══════════════════════════════════════════════════════════════════════
# 4. TIMELINE
# ═══════════════════════════════════════════════════════════════════════

def build_timeline(patient_dir, summary, file_order):
    """
    Load EDF files in CHRONOLOGICAL order (from summary file).
    Critical: glob sorts alphabetically → chb02_16+.edf BEFORE chb02_16.edf
    but summary lists them chronologically.
    """
    all_edfs = {os.path.basename(p): p
                for p in glob.glob(os.path.join(patient_dir, "*.edf"))}

    ordered_paths = []
    for fname in file_order:
        if fname in all_edfs:
            ordered_paths.append(all_edfs.pop(fname))
    for p in sorted(all_edfs.values()):
        ordered_paths.append(p)

    if not ordered_paths:
        raise FileNotFoundError(f"No .edf in {patient_dir}")

    file_list, g_seizures = [], []
    cum, skipped = 0, 0

    print(f"\n  Loading {len(ordered_paths)} EDF files (summary order)...")
    t0 = time.time()

    for pi, p in enumerate(ordered_paths):
        fname = os.path.basename(p)
        try:
            data, ch_names, fs = read_edf(p)
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}")
            skipped += 1
            continue

        n_samp = data.shape[1]
        file_list.append(dict(path=p, name=fname, start=cum, end=cum+n_samp,
                              n_samples=n_samp, ch_names=ch_names))

        if fname in summary:
            for sz_s, sz_e in summary[fname]:
                g_seizures.append(dict(onset=cum+sz_s*FS, offset=cum+sz_e*FS,
                                       file=fname, local_s=sz_s, local_e=sz_e))
        cum += n_samp

        if (pi+1) % 10 == 0 or pi+1 == len(ordered_paths):
            print(f"    {pi+1}/{len(ordered_paths)} ({time.time()-t0:.1f}s)", end="\r")

    print(f"    {len(file_list)} loaded, {skipped} skipped ({time.time()-t0:.1f}s)   ")
    g_seizures.sort(key=lambda x: x["onset"])
    return file_list, g_seizures, cum


def extract_segments_streaming(file_list, region_start, region_end, seg_samp, step_samp):
    """
    Memory-efficient segment extraction — kết quả GIỐNG HỆT load_range cũ.

    Thay vì load cả region (có thể hàng chục GB) → chỉ load TỐI ĐA 2 files
    cùng lúc (cho segment nằm trên ranh giới 2 files).

    Logic:
      1. Tính trước TẤT CẢ vị trí segment (global sample coords)
      2. Với mỗi segment, xác định nó thuộc file nào
      3. Load file đó, cắt segment, giải phóng
      4. Nếu segment nằm trên 2 files → load cả 2, ghép, cắt

    Returns: list of (22, seg_samp) arrays
    """
    # ── Bước 1: Tính trước tất cả segment positions ──
    seg_positions = []  # list of global_start for each segment
    pos = region_start
    while pos + seg_samp <= region_end:
        seg_positions.append(pos)
        pos += step_samp

    if not seg_positions:
        return []

    # ── Bước 2: Xây lookup file → global range ──
    # Chỉ giữ files overlap với region
    relevant_files = []
    for f in file_list:
        if f["end"] > region_start and f["start"] < region_end:
            relevant_files.append(f)

    if not relevant_files:
        return []

    # ── Bước 3: Extract từng segment ──
    segments = []
    cache = {}  # {file_path: (d22, f_info)} — cache tối đa 2 files

    def get_file_data(f_info):
        """Load file data, sử dụng cache để tránh đọc lại."""
        key = f_info["path"]
        if key not in cache:
            # Giữ cache tối đa 2 files (đủ cho cross-boundary)
            if len(cache) >= 2:
                # Xóa file cũ nhất
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            try:
                data, ch_names, _ = read_edf(f_info["path"])
                d22, _ = select_22ch(data, ch_names)
                del data  # Giải phóng raw data ngay
                cache[key] = d22
            except Exception as e:
                cache[key] = None
        return cache[key]

    def find_file(global_pos):
        """Tìm file chứa vị trí global_pos."""
        for f in relevant_files:
            if f["start"] <= global_pos < f["end"]:
                return f
        return None

    for seg_start in seg_positions:
        seg_end = seg_start + seg_samp

        f1 = find_file(seg_start)
        if f1 is None:
            continue

        # Case A: Segment nằm hoàn toàn trong 1 file
        if seg_end <= f1["end"]:
            d22 = get_file_data(f1)
            if d22 is None:
                continue
            local_s = seg_start - f1["start"]
            local_e = local_s + seg_samp
            if local_e <= d22.shape[1]:
                segments.append(d22[:, local_s:local_e].copy())

        # Case B: Segment nằm trên ranh giới 2 files
        else:
            f2 = find_file(seg_end - 1)  # File chứa sample cuối
            if f2 is None or f2["path"] == f1["path"]:
                continue

            d22_a = get_file_data(f1)
            d22_b = get_file_data(f2)
            if d22_a is None or d22_b is None:
                continue

            # Lấy phần cuối file 1 + phần đầu file 2
            part_a = d22_a[:, (seg_start - f1["start"]):]
            needed_b = seg_samp - part_a.shape[1]
            part_b = d22_b[:, :needed_b]

            if part_a.shape[1] + part_b.shape[1] == seg_samp:
                segments.append(np.concatenate([part_a, part_b], axis=1).copy())

    # Giải phóng cache
    cache.clear()

    return segments


# ═══════════════════════════════════════════════════════════════════════
# 5. FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def process_patient(patient_dir, seg_sec=SEG_SEC):
    pid = os.path.basename(patient_dir)
    seg_samp = seg_sec * FS

    print(f"\n{'═'*65}")
    print(f"  PATIENT: {pid.upper()}")
    print(f"{'═'*65}")

    summary, file_order = parse_summary(patient_dir)
    file_list, seizures, total_samp = build_timeline(patient_dir, summary, file_order)
    total_h = total_samp / FS / 3600

    print(f"\n  Recording: {total_h:.1f}h | Seizures: {len(seizures)}")
    for i, sz in enumerate(seizures):
        print(f"    #{i+1:2d}  {sz['onset']/FS:>10.0f}s – {sz['offset']/FS:>10.0f}s  "
              f"(dur={( sz['offset']-sz['onset'])/FS:.0f}s)  {sz['file']}")

    if len(seizures) < 2:
        print(f"\n  [ERROR] Need ≥2 seizures for LOSO")
        return None

    # ── Identify leading seizures ──
    # Leading = ≥30min from previous seizure
    # Paper Table I counts only leading seizures (CHB14: 5, not 8)
    # LOSO folds = leading seizures only (each fold has preictal in test)
    is_lead = [True]
    for i in range(1, len(seizures)):
        gap_samples = seizures[i]["onset"] - seizures[i-1]["offset"]
        is_lead.append(gap_samples >= LEAD_GAP_SEC * FS)

    leads = [sz for sz, fl in zip(seizures, is_lead) if fl]
    n_sz = len(leads)  # LOSO folds = leading seizures only

    print(f"\n  Total seizures: {len(seizures)} | Leading: {n_sz} → {n_sz} LOSO folds")
    for i, (sz, fl) in enumerate(zip(seizures, is_lead)):
        print(f"    #{i+1:2d}  {'LEADING ✓' if fl else 'non-leading (skip for LOSO)'}")

    if n_sz < 2:
        print(f"  [ERROR] Need ≥2 leading seizures for LOSO")
        return None

    # ── Preictal regions: 30min before onset, minus 5min intervention ──
    # Only for leading seizures. Avoid overlap with previous seizure's ictal.
    pre_dur = PREICTAL_SEC * FS
    interv = INTERV_SEC * FS
    gap = GAP_SEC * FS

    pre_regions = []
    for li, sz in enumerate(leads):
        # Find this seizure's index in the full seizures list
        full_idx = seizures.index(sz)

        ps = max(0, sz["onset"] - pre_dur)
        pe = sz["onset"] - interv

        # Avoid overlap with any previous seizure (including non-leading)
        if full_idx > 0:
            ps = max(ps, seizures[full_idx - 1]["offset"])

        if pe > ps + seg_samp:
            pre_regions.append(dict(start=ps, end=pe, lead_idx=li))

    # ── Interictal regions: 4h gap from ALL seizures (including non-leading) ──
    # This is safer: non-leading seizures are still real seizures
    inter_regions = []
    r_end = seizures[0]["onset"] - gap
    if r_end > seg_samp:
        inter_regions.append(dict(start=0, end=r_end))
    for i in range(len(seizures) - 1):
        rs = seizures[i]["offset"] + gap
        re_ = seizures[i+1]["onset"] - gap
        if re_ > rs + seg_samp:
            inter_regions.append(dict(start=rs, end=re_))
    rs = seizures[-1]["offset"] + gap
    if total_samp > rs + seg_samp:
        inter_regions.append(dict(start=rs, end=total_samp))

    total_pre_min = sum((r["end"]-r["start"])/FS/60 for r in pre_regions)
    total_inter_h = sum((r["end"]-r["start"])/FS/3600 for r in inter_regions)
    print(f"\n  Preictal:   {len(pre_regions)} regions ({total_pre_min:.1f} min)")
    for r in pre_regions:
        print(f"    lead #{r['lead_idx']+1}: [{r['start']/FS:.0f}s → {r['end']/FS:.0f}s] "
              f"({(r['end']-r['start'])/FS/60:.1f} min)")
    print(f"  Interictal: {len(inter_regions)} regions ({total_inter_h:.1f}h)")

    # Balancing — Paper formulas 1-3 (EXACT, no modification)
    # K = M/N, preictal step = S × K
    # This creates balanced classes even when preictal << interictal
    # CHB02: K=0.054 → 95% overlap → ~16K preictal ≈ ~16K interictal
    M = total_pre_min * 60
    N = total_inter_h * 3600
    K = M / N if N > 0 else 1.0
    pre_step = max(1, int(seg_sec * K * FS))

    overlap_pct = (1 - pre_step / seg_samp) * 100
    print(f"\n  K={K:.6f} | pre_step={pre_step} samples ({pre_step/FS:.4f}s) "
          f"| overlap={overlap_pct:.1f}%")

    # ── Phase 1: Count segments (no STFT, no RAM) ──
    print(f"\n  Counting segments...")

    pre_counts = []  # segments per preictal region
    for r in pre_regions:
        n = max(0, (r["end"] - r["start"] - seg_samp) // pre_step + 1)
        pre_counts.append(n)
    total_pre = sum(pre_counts)

    inter_counts = []
    for r in inter_regions:
        n = max(0, (r["end"] - r["start"] - seg_samp) // seg_samp + 1)
        inter_counts.append(n)
    total_inter = sum(inter_counts)

    pre_mb = total_pre * 22 * 9 * 114 * 4 / 1e6
    inter_mb = total_inter * 22 * 9 * 114 * 4 / 1e6
    print(f"  Estimated: preictal={total_pre} ({pre_mb:.0f}MB), "
          f"interictal={total_inter} ({inter_mb:.0f}MB)")

    # ── Phase 2: Pre-allocate + extract + STFT in-place ──
    print(f"\n  Extracting segments + STFT conversion...")
    # Tính tự động số mốc thời gian (sẽ = 39)
    nt = (seg_samp - NPERSEG) // (NPERSEG - NOVERLAP) + 1

    pre_specs = np.empty((total_pre, 22, nt, 114), dtype=np.float16)
    pre_ids = np.empty(total_pre, dtype=np.int32)
    idx = 0
    for ri, r in enumerate(pre_regions):
        print(f"  Preictal lead#{r['lead_idx']+1}...", end=" ", flush=True)
        segs = extract_segments_streaming(
            file_list, r["start"], r["end"], seg_samp, pre_step)
        for s in segs:
            pre_specs[idx] = stft_22ch(s)
            pre_ids[idx] = r["lead_idx"]
            idx += 1
        print(f"{len(segs)} segs → spectrograms")
        del segs
    pre_specs = pre_specs[:idx]  # trim if actual < estimated
    pre_ids = pre_ids[:idx]

    inter_specs = np.empty((total_inter, 22, nt, 114), dtype=np.float16)
    idx = 0
    for ri, r in enumerate(inter_regions):
        print(f"  Interictal {ri+1}/{len(inter_regions)}...", end=" ", flush=True)
        segs = extract_segments_streaming(
            file_list, r["start"], r["end"], seg_samp, seg_samp)
        cnt = 0
        for s in segs:
            inter_specs[idx] = stft_22ch(s)
            idx += 1; cnt += 1
            if cnt % 500 == 0:
                print(f"{cnt}", end=" ", flush=True)
        print(f"{cnt} segs → spectrograms")
        del segs
    inter_specs = inter_specs[:idx]

    n_inter = len(inter_specs)
    grp_sz = max(1, n_inter // n_sz)
    inter_grp = np.array([min(i//grp_sz, n_sz-1) for i in range(n_inter)])

    print(f"\n  ▸ Preictal:   {pre_specs.shape} ({pre_specs.nbytes/1e6:.1f}MB)")
    print(f"  ▸ Interictal: {inter_specs.shape} ({inter_specs.nbytes/1e6:.1f}MB)")
    print(f"  ▸ Folds:      {n_sz} (split during training)")

    return dict(pre_specs=pre_specs, inter_specs=inter_specs,
                pre_ids=pre_ids, inter_grp=inter_grp,
                n_sz=n_sz, pid=pid)


def save_all(out_dir, r):
    pid = r["pid"]
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{pid}.npz")
    temp_path = os.path.join(out_dir, f"{pid}_temp.npz")

    # 1. Ghi thẳng ra đĩa
    np.savez(temp_path,
             preictal=r["pre_specs"],
             interictal=r["inter_specs"],
             preictal_seizure_ids=r["pre_ids"],
             interictal_group_ids=r["inter_grp"],
             n_folds=np.array(r["n_sz"]))

    # Đảm bảo file cũ không còn tồn tại
    if os.path.exists(out_path):
        try:
            os.remove(out_path)
        except PermissionError:
            print(f"    [CẢNH BÁO] Không thể xóa file cũ {out_path} do đang bị khóa.")

    # 2. VÒNG LẶP CHỜ WINDOWS NHẢ FILE (Tránh WinError 32)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.rename(temp_path, out_path)
            break  # Nếu đổi tên thành công thì thoát vòng lặp
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"    [ĐANG CHỜ] File đang bị Windows khóa (Antivirus/Cloud). Thử lại sau 2 giây... ({attempt+1}/{max_retries})")
                time.sleep(2) # Chờ 2 giây rồi thử lại
            else:
                print("    [LỖI] Windows khóa file quá lâu. Đổi tên thất bại.")
                raise e # Hết số lần thử mà vẫn lỗi thì đành chịu

    file_size = os.path.getsize(out_path) / 1e6
    raw_mb = (r["pre_specs"].nbytes + r["inter_specs"].nbytes) / 1e6

    print(f"\n  ✓ Saved → {out_path}")
    print(f"    Preictal:   {r['pre_specs'].shape}")
    print(f"    Interictal: {r['inter_specs'].shape}")
    print(f"    Folds:      {r['n_sz']}")
    print(f"    Size:       {file_size:.1f} MB (raw {raw_mb:.1f} MB)")

def verify():
    print("═" * 65)
    print("  STFT DIMENSION VERIFICATION")
    print("═" * 65)
    n = SEG_SEC * FS
    nt = (n - NPERSEG) // (NPERSEG - NOVERLAP) + 1
    nf = NFFT//2+1 - len(REMOVE_BINS)
    spec = stft_22ch(np.random.randn(22, n))
    ok = spec.shape == (22, 39, 114)
    print(f"  {SEG_SEC}s × {FS}Hz = {n} samples")
    print(f"  STFT: nperseg={NPERSEG}, noverlap={NOVERLAP}, nfft={NFFT}")
    print(f"  → time={nt}, freq={nf}")
    print(f"  Output:   {spec.shape}")
    print(f"  Expected: (22, 39, 114)")
    print(f"  {'✓ MATCH!' if ok else '✗ MISMATCH!'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"D:/archive/chb-mit-scalp-eeg-database-1.0.0")
    p.add_argument("--output_dir", default="D:/archive/processed_data(5)")
    p.add_argument("--patients", nargs="+", default=[
    "chb01","chb02","chb03","chb05","chb09","chb10",
    "chb13","chb14","chb18","chb19","chb20","chb21","chb23"])
    p.add_argument("--segment_length", type=int, default=5, choices=[5,15,30])
    p.add_argument("--verify_only", action="store_true")
    args = p.parse_args()

    if not verify():
        return

    if args.verify_only:
        return

    global SEG_SEC
    SEG_SEC = args.segment_length
    os.makedirs(args.output_dir, exist_ok=True)

    for pid in args.patients:
        pdir = os.path.join(args.data_dir, pid)
        if not os.path.isdir(pdir):
            print(f"\n  [ERROR] Không tìm thấy thư mục: {pdir}")
            continue
            
        # ==============================================================
        # CƠ CHẾ RESUME: KIỂM TRA BỆNH NHÂN ĐÃ HOÀN THÀNH CHƯA
        # ==============================================================
        out_file = os.path.join(args.output_dir, f"{pid}.npz")
        if os.path.exists(out_file):
            print(f"\n{'═'*65}")
            print(f"  [SKIP] Bệnh nhân {pid.upper()} đã hoàn thành trước đó.")
            print(f"         File tồn tại: {out_file}")
            print(f"{'═'*65}")
            continue  # Bỏ qua, chạy sang bệnh nhân tiếp theo!
        # ==============================================================

        t0 = time.time()
        try:
            result = process_patient(pdir, args.segment_length)
            if result:
                save_all(args.output_dir, result)
        except Exception as e:
            print(f"\n  [ERROR] {pid}: {e}")
            import traceback
            traceback.print_exc()
        print(f"  Time: {time.time()-t0:.0f}s")

    print(f"\n{'═'*65}")
    print(f"  XONG — Output: {os.path.abspath(args.output_dir)}")
    print(f"{'═'*65}")

if __name__ == "__main__":
    main()