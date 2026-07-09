#!/usr/bin/env python3
import subprocess
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_sub_program(name, command):
    """
    线程池里的核心工作单元：负责拉起子程序并实时监控其生死
    """
    print(f"🚀 [系统启动] 正在通过线程池部署: 【{name}】...")
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(proc.stdout.readline, ''):
            if line.strip():
                print(f"[{name}] {line.strip()}")
                
        proc.wait()
        return f"✅ 【{name}】已正常退出，状态码: {proc.returncode}"
    except Exception as e:
        return f"❌ 【{name}】运行期间发生异常: {e}"

def main():
    print("==================================================")
    print(" 🧠 RK3588 多模态中枢 - 线程池并发架构总控 v2.0")
    print("==================================================")
    
    cmd = input("👉 请输入指令激活多模态系统 (输入 000 一键全开): ").strip()
    
    if cmd == "000":
        TASKS_CONFIG = {
            "脑电模块 (egg.py)": ["python3", "egg.py"],
            "视觉模块 (AL_GONGER.py)": ["python3", "AL_GONGER.py"]
        }
        
        max_workers = max(len(TASKS_CONFIG), 4)
        print(f"\n⚡ 正在初始化核心线程池 (最大并发数: {max_workers})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for name, command in TASKS_CONFIG.items():
                future = executor.submit(run_sub_program, name, command)
                futures[future] = name
                
            print("==================================================")
            print("💡 系统全线轰鸣！按 Ctrl+C 可一键撤回并安全关闭所有模块。")
            print("==================================================\n")
            
            try:
                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    print(f"\n⚠️ 线程池警报: {result}")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n👋 收到用户中断信号 (Ctrl+C)，正在安全回收线程池...")
            finally:
                print("🛑 正在强制剥离并清理残留后台进程...")
                subprocess.Popen("pkill -f egg.py", shell=True)
                subprocess.Popen("pkill -f AL_GONGER.py", shell=True)
                print("✅ 线程池释放完毕，板子环境已洗净。")
    else:
        print("❌ 指令错误，线程池拒绝初始化。")

if __name__ == "__main__":
    main()
