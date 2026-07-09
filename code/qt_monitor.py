import sys, requests, cv2, numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
import csv, os
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import rcParams
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ===== 中文字体配置 =====
import matplotlib.font_manager as fm
font_paths = [
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
]
for path in font_paths:
    if os.path.exists(path):
        fm.fontManager.addfont(path)
        break
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BOARD_IP = "127.0.0.1"
DESKTOP = os.path.expanduser("~/Desktop")

# ===== 脑电数据滑动平均滤波器 =====
class MovingAverageFilter:
    def __init__(self, window=5):
        self.window = window
        self.buffer = []

    def update(self, value):
        self.buffer.append(value)
        if len(self.buffer) > self.window:
            self.buffer.pop(0)
        return np.mean(self.buffer)

eeg_filter = MovingAverageFilter(window=5)

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    def run(self):
        bytes_data = b''
        try:
            r = requests.get(f"http://{BOARD_IP}:8080/stream", stream=True, timeout=10)
            for chunk in r.iter_content(chunk_size=4096):
                bytes_data += chunk
                a, b = bytes_data.find(b'\xff\xd8'), bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    frame = cv2.imdecode(np.frombuffer(bytes_data[a:b+2], dtype=np.uint8), 1)
                    if frame is not None:
                        self.frame_ready.emit(frame)
                    bytes_data = bytes_data[b+2:]
        except:
            pass

class QtMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Smart Monitor - Pro Edition")
        self.setStyleSheet("QMainWindow { background-color: #0F172A; }")
        self.resize(1100, 780)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        title = QLabel("RK3588 INTELLIGENT HEALTH & BRAIN-COMPUTER INTERFACE ANALYSIS SYSTEM")
        title.setStyleSheet("color: #38BDF8; font: bold 18px 'Consolas'; letter-spacing: 3px;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        content_layout = QHBoxLayout()

        self.v_label = QLabel()
        self.v_label.setFixedSize(640, 480)
        self.v_label.setStyleSheet("border: 2px solid #38BDF8; border-radius: 10px; background-color: #000;")
        content_layout.addWidget(self.v_label)

        right_panel = QVBoxLayout()

        # 分数卡片
        score_card = QFrame()
        score_card.setStyleSheet("background-color: #1E293B; border-radius: 15px; border: 1px solid #334155;")
        sc_layout = QVBoxLayout(score_card)
        sc_title = QLabel("POSTURE SCORE & EEG MENTAL STATE")
        sc_title.setStyleSheet("color: #94A3B8; font: bold 12px;")
        sc_layout.addWidget(sc_title)

        self.score_box = QLabel("100")
        self.score_box.setAlignment(Qt.AlignCenter)
        self.score_box.setStyleSheet("color: #4ADE80; font: bold 72px 'Digital-7';")
        sc_layout.addWidget(self.score_box)

        self.eeg_state_label = QLabel("🧠 专注中")
        self.eeg_state_label.setAlignment(Qt.AlignCenter)
        self.eeg_state_label.setStyleSheet("color: #FBBF24; font: bold 18px;")
        sc_layout.addWidget(self.eeg_state_label)

        right_panel.addWidget(score_card)

        # 曲线1
        self.graph1 = pg.PlotWidget()
        self.graph1.setBackground('#1E293B')
        self.graph1.setTitle("颈椎偏离率实时监测", color='#94A3B8', size='12px')
        self.graph1.showGrid(x=False, y=True, alpha=0.3)
        self.graph1.setLabel('left', '偏离率', color='#94A3B8', size='10px')
        self.graph1.setLabel('bottom', '时间 (帧)', color='#94A3B8', size='10px')
        self.curve_posture = self.graph1.plot(pen=pg.mkPen(color='#38BDF8', width=3))
        self.data_posture = [0]*60
        right_panel.addWidget(self.graph1)

        # 曲线2（脑电，使用平滑数据）
        self.graph2 = pg.PlotWidget()
        self.graph2.setBackground('#1E293B')
        self.graph2.setTitle("脑电专注度走势（平滑）", color='#94A3B8', size='12px')
        self.graph2.showGrid(x=False, y=True, alpha=0.3)
        self.graph2.setLabel('left', '专注度指数', color='#94A3B8', size='10px')
        self.graph2.setLabel('bottom', '时间 (帧)', color='#94A3B8', size='10px')
        self.curve_eeg = self.graph2.plot(pen=pg.mkPen(color='#FBBF24', width=3))
        self.data_eeg = [1.0]*60
        ticks = [list(zip(range(0, 61, 10), [str(i) for i in range(0, 61, 10)]))]
        self.graph2.getAxis('bottom').setTicks(ticks)
        right_panel.addWidget(self.graph2)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.calib_btn = QPushButton("🧘 校准坐姿 (保持10秒)")
        self.calib_btn.setStyleSheet("background-color: #3B82F6; color: white; font: bold 14px; border-radius: 8px; padding: 8px;")
        self.calib_btn.clicked.connect(self.calibrate_posture)
        btn_layout.addWidget(self.calib_btn)

        # 生成报告按钮
        self.save_btn = QPushButton("📊 生成学习报告")
        self.save_btn.setStyleSheet("background-color: #10B981; color: white; font: bold 14px; border-radius: 8px; padding: 8px;")
        self.save_btn.clicked.connect(self.save_chart)
        btn_layout.addWidget(self.save_btn)

        right_panel.addLayout(btn_layout)

        self.status_label = QLabel("📝 记录中...")
        self.status_label.setStyleSheet("color: #6B7280; font: 12px;")
        right_panel.addWidget(self.status_label)

        self.llm_label = QLabel("🤖 智能关怀：等待中...")
        self.llm_label.setStyleSheet("color: #FBBF24; font: 14px; background-color: #1E293B; border-radius: 8px; padding: 8px;")
        self.llm_label.setWordWrap(True)
        self.llm_label.setFixedHeight(60)
        right_panel.addWidget(self.llm_label)

        content_layout.addLayout(right_panel)
        main_layout.addLayout(content_layout)

        # 数据记录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(DESKTOP, f"study_data_{timestamp}.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'rat', 'score', 'eeg_state', 'eeg_score'])
        self.csv_file.flush()
        self.record_counter = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

        self.vt = VideoThread()
        self.vt.frame_ready.connect(self.update_video)
        self.vt.start()

    def calibrate_posture(self):
        try:
            url = f"http://{BOARD_IP}:8080/calibrate"
            resp = requests.get(url, timeout=10)
            if "成功" in resp.text:
                QMessageBox.information(self, "校准", "✅ 校准成功！请重启开发板上的程序。")
            else:
                QMessageBox.warning(self, "校准", "❌ 校准失败，请保持标准坐姿10秒。")
        except Exception as e:
            QMessageBox.critical(self, "网络错误", f"无法连接开发板：{e}")

    def update_data(self):
        try:
            r = requests.get(f"http://{BOARD_IP}:8080/status", timeout=0.5).json()
            score = int(r['sc'])
            rat = r['rat']
            eeg_state = r.get('eeg_st', 'NORMAL')
            eeg_sc = r.get('eeg_sc', 1.0)

            # 应用滑动平均滤波使脑电曲线更平滑
            smooth_eeg = eeg_filter.update(eeg_sc)

            self.score_box.setText(str(score))
            color = "#F87171" if score < 70 else "#4ADE80"
            self.score_box.setStyleSheet(f"color: {color}; font: bold 72px 'Digital-7';")

            state_map = {'FOCUS': '🧠 专注中', 'NORMAL': '🧠 平稳', 'RELAX': '🧠 放松/疲劳'}
            self.eeg_state_label.setText(state_map.get(eeg_state, '🧠 平稳'))

            self.data_posture.append(rat)
            self.data_posture.pop(0)
            self.curve_posture.setData(self.data_posture)

            # 使用平滑后的脑电数据
            self.data_eeg.append(smooth_eeg)
            self.data_eeg.pop(0)
            self.curve_eeg.setData(self.data_eeg)

            self.record_counter += 1
            if self.record_counter % 5 == 0:
                self.csv_writer.writerow([
                    datetime.now().isoformat(),
                    round(rat, 4),
                    score,
                    eeg_state,
                    round(eeg_sc, 4)  # 记录原始数据
                ])
                self.csv_file.flush()

            self.status_label.setText(f"📝 记录中 (已记录 {self.record_counter} 条)")

        except:
            pass

        try:
            resp = requests.get(f"http://{BOARD_IP}:8080/last_message", timeout=1)
            if resp.status_code == 200:
                msg = resp.json().get("message", "")
                if msg:
                    self.llm_label.setText(f"🤖 智能关怀：{msg}")
        except:
            pass

    def save_chart(self):
        """生成学习报告图"""
        if not HAS_MPL:
            QMessageBox.warning(self, "缺少依赖", "请安装: pip3 install pandas matplotlib --user")
            return

        try:
            if not os.path.exists(self.csv_path):
                QMessageBox.warning(self, "无数据", "暂无数据文件")
                return

            df = pd.read_csv(self.csv_path)
            if df.empty:
                QMessageBox.warning(self, "无数据", "数据文件为空")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            if df.empty:
                QMessageBox.warning(self, "无数据", "时间戳解析失败")
                return

            timestamps = df['timestamp'].values
            rat_vals = df['rat'].values.astype(np.float64)
            score_vals = df['score'].values.astype(np.float64)
            eeg_vals = df['eeg_score'].values.astype(np.float64)
            eeg_states = df['eeg_state'].values

            avg_score = np.mean(score_vals)
            avg_rat = np.mean(rat_vals)
            avg_eeg = np.mean(eeg_vals)
            focus_pct = np.sum(eeg_states == 'FOCUS') / len(eeg_states) * 100
            relax_pct = np.sum(eeg_states == 'RELAX') / len(eeg_states) * 100
            normal_pct = np.sum(eeg_states == 'NORMAL') / len(eeg_states) * 100
            total_minutes = len(df) / 60

            # 分析文字
            if focus_pct > 50:
                state_text = "本次学习专注度较高，建议继续保持良好习惯。"
            elif relax_pct > 40:
                state_text = "本次学习偏向放松状态，建议适当增加专注训练。"
            else:
                state_text = "本次学习状态平稳，建议结合自身节奏调整学习强度。"

            if avg_score < 60:
                state_text += " 健康分偏低，建议调整坐姿和休息节奏。"
            elif avg_score < 80:
                state_text += " 健康分良好，仍有提升空间。"
            else:
                state_text += " 健康分优秀，继续保持！"

            start_time = df['timestamp'].min().strftime('%H:%M')
            end_time = df['timestamp'].max().strftime('%H:%M')

            # 设置字体大小
            plt.rcParams['font.size'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12
            plt.rcParams['legend.fontsize'] = 12

            fig = plt.figure(figsize=(16, 10), facecolor='#0F172A')
            fig.patch.set_facecolor('#0F172A')

            fig.suptitle('学习阶段报告', fontsize=26, color='white', fontweight='bold', y=0.98)

            ax_time = fig.add_axes([0.06, 0.90, 0.88, 0.04])
            ax_time.axis('off')
            ax_time.text(0.5, 0.5, f'时间: {start_time} - {end_time}  |  总时长: {total_minutes:.0f} 分钟',
                         color='#94A3B8', fontsize=14, ha='center', va='center')

            ax_metrics = fig.add_axes([0.06, 0.82, 0.88, 0.10])
            ax_metrics.axis('off')
            metrics_text = (
                f'总得分: {avg_score:.0f}    |    '
                f'偏移率: {avg_rat:.3f}    |    '
                f'脑电指数: {avg_eeg:.3f}'
            )
            ax_metrics.text(0.5, 0.5, metrics_text,
                            color='white', fontsize=24, ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.6', facecolor='#1E293B', edgecolor='#334155'))

            ax_rat = fig.add_axes([0.06, 0.48, 0.55, 0.28])
            ax_rat.plot(timestamps, rat_vals, color='#38BDF8', linewidth=2)
            ax_rat.set_ylabel('偏移率', color='#38BDF8', fontsize=14)
            ax_rat.tick_params(axis='y', labelcolor='#38BDF8')
            ax_rat.grid(True, alpha=0.15, color='#334155')
            ax_rat.set_title('颈椎偏移率趋势', color='#94A3B8', fontsize=16)
            ax_rat.set_facecolor('#0F172A')
            for spine in ax_rat.spines.values():
                spine.set_color('#334155')

            ax_eeg = fig.add_axes([0.06, 0.16, 0.55, 0.28])
            ax_eeg.plot(timestamps, eeg_vals, color='#FBBF24', linewidth=2)
            ax_eeg.set_ylabel('脑电指数', color='#FBBF24', fontsize=14)
            ax_eeg.tick_params(axis='y', labelcolor='#FBBF24')
            ax_eeg.grid(True, alpha=0.15, color='#334155')
            ax_eeg.set_xlabel('时间', color='#94A3B8', fontsize=14)
            ax_eeg.set_title('脑电专注度趋势', color='#94A3B8', fontsize=16)
            ax_eeg.set_facecolor('#0F172A')
            for spine in ax_eeg.spines.values():
                spine.set_color('#334155')

            ax_pie = fig.add_axes([0.73, 0.62, 0.26, 0.24])
            labels = ['专注', '平稳', '放松']
            sizes = [focus_pct, normal_pct, relax_pct]
            colors = ['#4ADE80', '#FBBF24', '#38BDF8']
            wedges, texts, autotexts = ax_pie.pie(
                sizes, labels=labels, colors=colors, autopct='%1.0f%%',
                startangle=90, textprops={'fontsize': 16, 'color': 'white'}
            )
            ax_pie.set_title('精神状态分布', color='#94A3B8', fontsize=18)

            ax_analysis = fig.add_axes([0.68, 0.46, 0.26, 0.12])
            ax_analysis.axis('off')
            ax_analysis.text(0.02, 0.95, '分析', color='#94A3B8', fontsize=15, fontweight='bold', ha='left', va='top')
            ax_analysis.text(0.02, 0.70, state_text, color='#FBBF24', fontsize=15, ha='left', va='top', wrap=True)

            ax_box = fig.add_axes([0.68, 0.12, 0.26, 0.28])
            box_data = [eeg_vals]
            bp = ax_box.boxplot(box_data, vert=True, patch_artist=True,
                                labels=['脑电指数'],
                                boxprops=dict(facecolor='#1E293B', color='#FBBF24'),
                                whiskerprops=dict(color='#FBBF24'),
                                capprops=dict(color='#FBBF24'),
                                medianprops=dict(color='#4ADE80', linewidth=2),
                                flierprops=dict(marker='o', markerfacecolor='#FBBF24', markersize=6))
            ax_box.set_title('脑电指数分布', color='#94A3B8', fontsize=16)
            ax_box.set_ylabel('指数值', color='#94A3B8', fontsize=14)
            ax_box.tick_params(axis='y', labelcolor='#94A3B8')
            ax_box.set_facecolor('#0F172A')
            for spine in ax_box.spines.values():
                spine.set_color('#334155')

            ax_info = fig.add_axes([0.68, 0.06, 0.26, 0.04])
            ax_info.axis('off')
            ax_info.text(0.5, 0.5, f'数据点数: {len(df)}',
                         color='#6B7280', fontsize=14, ha='center', va='center')

            save_path = os.path.join(DESKTOP, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#0F172A')
            plt.close()

            QMessageBox.information(self, "报告生成", f"报告已保存到桌面:\n{save_path}")
            self.status_label.setText(f"报告已生成: {os.path.basename(save_path)}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"{str(e)}")

    def update_video(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        qimg = QImage(img.data, w, h, w*ch, QImage.Format_RGB888)
        self.v_label.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.csv_file.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QtMonitor()
    win.show()
    sys.exit(app.exec_())
