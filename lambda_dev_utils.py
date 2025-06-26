#!/usr/bin/env python3
"""
Lambda Cloud Development Utilities
Lambda Cloud環境での開発・学習実行を支援するユーティリティスクリプト
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# SSH設定
SSH_KEY = "~/.ssh/lambda_cloud_key"
LAMBDA_IP = "150.136.47.58"
LAMBDA_USER = "ubuntu"
CODE_PATH = "/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux"
VENV_PATH = "/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv"

def run_ssh_command(command, use_tmux=False, session_name=None, detach=False):
    """SSH経由でコマンドを実行"""
    ssh_base = f"ssh -i {SSH_KEY} {LAMBDA_USER}@{LAMBDA_IP}"
    
    if use_tmux:
        session_name = session_name or "lisa_dev"
        if detach:
            # バックグラウンドで tmux セッションを作成し、コマンドを実行
            tmux_cmd = f"tmux new-session -d -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {command}'"
        else:
            # tmux セッションにアタッチ
            tmux_cmd = f"tmux attach-session -t {session_name} || tmux new-session -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {command}'"
        
        full_command = f"{ssh_base} \"{tmux_cmd}\""
    else:
        full_command = f"{ssh_base} \"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {command}\""
    
    print(f"🚀 実行中: {full_command}")
    
    if detach:
        # バックグラウンド実行
        process = subprocess.Popen(full_command, shell=True)
        print(f"✅ バックグラウンドで実行開始 (PID: {process.pid})")
        return process
    else:
        # 前景実行
        return subprocess.run(full_command, shell=True)

def sync_code():
    """ローカルコードをLambda Cloudに同期"""
    print("📤 コードをLambda Cloudに同期中...")
    
    rsync_cmd = f"""rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='lambda_results' --exclude='runs' \
        ./ {LAMBDA_USER}@{LAMBDA_IP}:{CODE_PATH}/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ コード同期完了")
    else:
        print("❌ コード同期失敗")
    return result.returncode == 0

def sync_results():
    """Lambda Cloudから結果をローカルに同期"""
    print("📥 結果をLambda Cloudから取得中...")
    
    # ローカルの結果ディレクトリを作成
    os.makedirs("lambda_results", exist_ok=True)
    
    rsync_cmd = f"""rsync -avz {LAMBDA_USER}@{LAMBDA_IP}:/lambda/nfs/lisa-gemma-project-fs/artifacts/ \
        ./lambda_results/"""
    
    result = subprocess.run(rsync_cmd, shell=True)
    if result.returncode == 0:
        print("✅ 結果取得完了")
    else:
        print("❌ 結果取得失敗")
    return result.returncode == 0

def list_tmux_sessions():
    """tmuxセッション一覧を表示"""
    print("📋 Lambda Cloud上のtmuxセッション:")
    run_ssh_command("tmux list-sessions", use_tmux=False)

def attach_tmux_session(session_name="lisa_dev"):
    """tmuxセッションにアタッチ"""
    print(f"🔗 tmuxセッション '{session_name}' にアタッチ中...")
    run_ssh_command("", use_tmux=True, session_name=session_name)

def kill_tmux_session(session_name="lisa_dev"):
    """tmuxセッションを終了"""
    print(f"🛑 tmuxセッション '{session_name}' を終了中...")
    run_ssh_command(f"tmux kill-session -t {session_name}", use_tmux=False)

def check_gpu_status():
    """GPU使用状況を確認"""
    print("🖥️  GPU使用状況:")
    run_ssh_command("nvidia-smi", use_tmux=False)

def emergency_stop():
    """緊急停止: 全てのPythonプロセスを停止"""
    print("🚨 緊急停止: 全Pythonプロセスを終了中...")
    
    # 確認
    response = input("⚠️  本当に全てのPythonプロセスを停止しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ キャンセルされました")
        return
    
    # 全Pythonプロセスを終了
    run_ssh_command("pkill -f python", use_tmux=False)
    print("✅ 緊急停止完了")

def start_training(script_name, experiment_name=None, use_tmux=True):
    """学習開始"""
    if experiment_name is None:
        experiment_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🏋️  学習開始: {script_name} (実験名: {experiment_name})")
    
    # まずコードを同期
    if not sync_code():
        print("❌ コード同期に失敗しました")
        return False
    
    # 学習コマンドを構築
    training_cmd = f"python {script_name} --experiment_name {experiment_name}"
    
    if use_tmux:
        print("📺 tmuxセッションで学習を開始します...")
        print("   SSH接続が切れても学習は継続されます")
        print(f"   再接続するには: python {__file__} attach")
        
        # tmuxでバックグラウンド実行
        process = run_ssh_command(training_cmd, use_tmux=True, 
                                session_name=f"training_{experiment_name}", 
                                detach=True)
        
        time.sleep(2)  # tmuxセッション開始を待つ
        print(f"✅ 学習開始完了 (tmuxセッション: training_{experiment_name})")
        return True
    else:
        # 前景実行（非推奨）
        print("⚠️  前景実行での学習開始 (SSH切断で停止します)")
        result = run_ssh_command(training_cmd, use_tmux=False)
        return result.returncode == 0

def monitor_training():
    """学習状況をモニタリング"""
    print("📊 学習状況モニタリング:")
    print("=" * 50)
    
    # GPU使用状況
    check_gpu_status()
    print()
    
    # tmuxセッション一覧
    list_tmux_sessions()
    print()
    
    # 最新のログファイルを表示
    print("📝 最新ログ:")
    run_ssh_command("find /lambda/nfs/lisa-gemma-project-fs/artifacts/logs -name '*.log' -type f -exec ls -lt {} + | head -5", use_tmux=False)

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("""
🚀 Lambda Cloud開発ユーティリティ

使用方法:
  python lambda_dev_utils.py <command> [options]

コマンド:
  sync              コードをLambda Cloudに同期
  results           結果をローカルに取得
  train <script>    学習開始 (tmuxセッション使用)
  monitor           学習状況モニタリング
  gpu               GPU使用状況確認
  tmux-list         tmuxセッション一覧
  attach [session]  tmuxセッションにアタッチ
  kill [session]    tmuxセッション終了
  emergency         緊急停止 (全Pythonプロセス終了)

例:
  python lambda_dev_utils.py sync
  python lambda_dev_utils.py train train_ds.py
  python lambda_dev_utils.py monitor
  python lambda_dev_utils.py attach training_20241226_143022
  python lambda_dev_utils.py emergency
        """)
        return
    
    command = sys.argv[1]
    
    if command == "sync":
        sync_code()
    
    elif command == "results":
        sync_results()
    
    elif command == "train":
        if len(sys.argv) < 3:
            print("❌ 学習スクリプト名を指定してください")
            return
        script_name = sys.argv[2]
        experiment_name = sys.argv[3] if len(sys.argv) > 3 else None
        start_training(script_name, experiment_name)
    
    elif command == "monitor":
        monitor_training()
    
    elif command == "gpu":
        check_gpu_status()
    
    elif command == "tmux-list":
        list_tmux_sessions()
    
    elif command == "attach":
        session_name = sys.argv[2] if len(sys.argv) > 2 else "lisa_dev"
        attach_tmux_session(session_name)
    
    elif command == "kill":
        session_name = sys.argv[2] if len(sys.argv) > 2 else "lisa_dev"
        kill_tmux_session(session_name)
    
    elif command == "emergency":
        emergency_stop()
    
    else:
        print(f"❌ 不明なコマンド: {command}")

if __name__ == "__main__":
    main() 