#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import socket
import serial
import threading
import sys
from scipy.signal import butter, lfilter, iirnotch

# ==================== 配置 ====================
FS = 228
WINDOW_SIZE = FS * 3
SLIDE_INTERVAL = 0.5
UDP_IP = "127.0.0.1"
UDP_PORT = 9999

SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyS9"
BAUDRATE = 115200

FRAME_HEADER = b'\x06\x04\x01'
FRAME_TAIL = b'\x00\x00'
ESCAPE_BYTE = 0x23
VREF = 3.3
ADC_MAX = 255

CHANNEL = "ch1"

# ==================== 状态判定参数 ====================
THETA_MIN, THETA_MAX = 4.0, 8.0
ALPHA_MIN, ALPHA_MAX = 8.0, 13.0
BETA_MIN,  BETA_MAX  = 16.0, 30.0

FOCUS_THRESHOLD  = 1.10
RELAX_THRESHOLD  = 0.85
FATIGUE_THRESHOLD = 2.0
STATE_HYSTERESIS  = 3
CALIBRATION_SAMPLES = 8

# ==================== 全局状态 ====================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
eeg_buffer = []
eeg_lock = threading.Lock()
baseline_score = None
calibration_pool = []
running = True
serial_connected = False
sample_count = 0
frame_count = 0

state_history = []
current_stable_state = "NORMAL"

b_notch, a_notch = iirnotch(50.0, 30.0, FS)
b_band, a_band = butter(6, [0.5, 45.0], btype='band', fs=FS)

def parse_frame(frame_bytes):
    if len(frame_bytes) < 5:
        return []
    data_part = frame_bytes[3:-2]
    values = []
    i = 0
    while i < len(data_part):
        if data_part[i] == ESCAPE_BYTE and i + 1 < len(data_part):
            values.append(data_part[i + 1])
            i += 2
        else:
            values.append(data_part[i])
            i += 1
    pairs = []
    for j in range(0, len(values) - 1, 2):
        pairs.append((values[j], values[j + 1]))
    return pairs

def serial_reader():
    global eeg_buffer, running, serial_connected, sample_count, frame_count
    rx_buf = bytearray()
    ser = None
    while running:
        if ser is None or not ser.is_open:
            try:
                ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
                print(f"[OK] 串口已打开: {SERIAL_PORT} ({BAUDRATE} bps)")
                serial_connected = True
                rx_buf.clear()
            except Exception as e:
                print(f"[错误] 无法打开串口 {SERIAL_PORT}: {e}")
                print(f"       检查: 1)设备是否连接 2)权限: sudo chmod 666 {SERIAL_PORT}")
                serial_connected = False
                time.sleep(2)
                continue
        try:
            data = ser.read(256)
            if not data:
                continue
            rx_buf.extend(data)
            samples = []
            consumed = 0
            while True:
                start = rx_buf.find(FRAME_HEADER, consumed)
                if start == -1:
                    consumed = len(rx_buf)
                    break
                end = rx_buf.find(FRAME_TAIL, start + 3)
                if end == -1:
                    consumed = start
                    break
                frame = bytes(rx_buf[start:end + 2])
                pairs = parse_frame(frame)
                frame_count += 1
                for ch1_raw, ch2_raw in pairs:
                    if CHANNEL == "ch1":
                        voltage = ch1_raw * VREF / ADC_MAX
                    elif CHANNEL == "ch2":
                        voltage = ch2_raw * VREF / ADC_MAX
                    else:
                        voltage = (ch1_raw + ch2_raw) / 2.0 * VREF / ADC_MAX
                    samples.append(voltage)
                    sample_count += 1
                consumed = end + 2
            if consumed > 0:
                del rx_buf[:consumed]
            if len(rx_buf) > 4096:
                del rx_buf[:2048]
            if samples:
                with eeg_lock:
                    eeg_buffer.extend(samples)
                    if len(eeg_buffer) > WINDOW_SIZE * 2:
                        eeg_buffer = eeg_buffer[-WINDOW_SIZE:]
        except Exception as e:
            print(f"[错误] 串口读取: {e}")
            serial_connected = False
            try:
                ser.close()
            except:
                pass
            ser = None
            time.sleep(1)

def analyze_brainwave_state(raw_signal):
    global baseline_score, calibration_pool, state_history, current_stable_state
    signal_arr = np.array(raw_signal, dtype=float)
    median = np.median(signal_arr)
    mad = np.median(np.abs(signal_arr - median)) + 1e-6
    threshold = 3.5 * mad
    signal_arr[np.abs(signal_arr - median) > threshold] = median
    signal_centered = signal_arr - np.mean(signal_arr)
    std_val = np.std(signal_centered)
    if std_val > 1e-6:
        signal_centered = signal_centered / std_val
    window = np.hanning(len(signal_centered))
    signal_windowed = signal_centered * window
    filtered = lfilter(b_notch, a_notch, signal_windowed)
    filtered = lfilter(b_band, a_band, filtered)
    fft_vals = np.abs(np.fft.rfft(filtered))
    fft_freqs = np.fft.rfftfreq(len(filtered), 1.0/FS)
    idx_total = (fft_freqs >= 0.5) & (fft_freqs <= 45.0)
    idx_theta = (fft_freqs >= THETA_MIN) & (fft_freqs <= THETA_MAX)
    idx_alpha = (fft_freqs >= ALPHA_MIN) & (fft_freqs <= ALPHA_MAX)
    idx_beta  = (fft_freqs >= BETA_MIN)  & (fft_freqs <= BETA_MAX)
    total_power = np.mean(fft_vals[idx_total]) + 1e-6
    theta_power = np.mean(fft_vals[idx_theta]) + 1e-6
    alpha_power = np.mean(fft_vals[idx_alpha]) + 1e-6
    beta_power  = np.mean(fft_vals[idx_beta])  + 1e-6
    theta_rel = theta_power / total_power
    alpha_rel = alpha_power / total_power
    beta_rel  = beta_power  / total_power
    focus_index = beta_rel / (alpha_rel + theta_rel + 1e-6)
    fatigue_index = theta_rel / (alpha_rel + 1e-6)
    if baseline_score is None:
        calibration_pool.append(focus_index)
        if len(calibration_pool) >= CALIBRATION_SAMPLES:
            baseline_score = np.median(calibration_pool)
            print(f"\n[校准完成] baseline={baseline_score:.4f} ({CALIBRATION_SAMPLES}样本, 中位数)")
            print(f"  Theta={theta_rel:.3f} Alpha={alpha_rel:.3f} Beta={beta_rel:.3f}\n")
        return "CALIBRATING", 1.0
    final_score = focus_index / baseline_score
    # 状态判定（基于原始 final_score）
    if final_score > FOCUS_THRESHOLD:
        raw_state = "FOCUS"
    elif final_score < RELAX_THRESHOLD or fatigue_index > FATIGUE_THRESHOLD:
        raw_state = "RELAX"
    else:
        raw_state = "NORMAL"
    state_history.append(raw_state)
    if len(state_history) > STATE_HYSTERESIS:
        state_history.pop(0)
    if len(state_history) == STATE_HYSTERESIS:
        if all(s == raw_state for s in state_history):
            current_stable_state = raw_state
    # 将最终分数裁剪到 [0, 2] 区间，避免 Qt 界面出现异常超高值
    final_score = np.clip(final_score, 0.0, 2.0)
    print(f"  θ={theta_rel:.3f} α={alpha_rel:.3f} β={beta_rel:.3f} "
          f"| focus={focus_index:.3f} fatigue={fatigue_index:.3f} "
          f"| score={final_score:.3f} → {raw_state} (稳定:{current_stable_state})")
    return current_stable_state, final_score

def main():
    global running
    print("=" * 60)
    print("  KS脑电实时检测 → UDP")
    print("=" * 60)
    print(f"  串口:     {SERIAL_PORT}")
    print(f"  采样率:   {FS}Hz  窗口: {WINDOW_SIZE}样本 ({WINDOW_SIZE/FS:.1f}s)")
    print(f"  分析间隔: {SLIDE_INTERVAL}s")
    print(f"  通道:     {CHANNEL}")
    print(f"  UDP:      {UDP_IP}:{UDP_PORT}")
    print(f"  按 Ctrl+C 退出")
    print("=" * 60)
    print()
    reader = threading.Thread(target=serial_reader, daemon=True)
    reader.start()
    print("[状态] 等待BLE数据... (确保 KS板已开机 + XY-MBO6BA已配对)")
    wait_start = time.time()
    while running:
        with eeg_lock:
            buf_len = len(eeg_buffer)
        if buf_len > 0:
            print(f"[OK] 收到数据! 开始检测\n")
            break
        if time.time() - wait_start > 10:
            conn = "已连接" if serial_connected else "未连接"
            print(f"[警告] 10秒无数据 | 串口:{SERIAL_PORT} {conn}")
            print(f"       检查: KS板开关 / XY-MBO6BA配对 / 串口路径")
            wait_start = time.time()
        time.sleep(0.5)
    last_output_time = time.time()
    try:
        while running:
            current_time = time.time()
            if current_time - last_output_time >= SLIDE_INTERVAL:
                with eeg_lock:
                    buf_copy = list(eeg_buffer)
                if len(buf_copy) >= WINDOW_SIZE:
                    state_str, score = analyze_brainwave_state(buf_copy[-WINDOW_SIZE:])
                    msg = f"{state_str},{score:.4f}"
                    try:
                        sock.sendto(msg.encode('utf-8'), (UDP_IP, UDP_PORT))
                        conn = "OK" if serial_connected else "XX"
                        print(f"[{time.strftime('%H:%M:%S')}] 发送: {msg}  "
                              f"样本:{sample_count} 帧:{frame_count} [{conn}]")
                    except Exception as e:
                        print(f"[错误] UDP发送失败: {e}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] 等待数据... "
                          f"{len(buf_copy)}/{WINDOW_SIZE}")
                last_output_time = current_time
            time.sleep(0.05)
    except KeyboardInterrupt:
        print(f"\n\n退出")
        running = False
        print(f"  总样本: {sample_count}  总帧: {frame_count}")

if __name__ == '__main__':
    main()
