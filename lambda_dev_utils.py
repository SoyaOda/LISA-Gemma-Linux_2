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
import json

# SSH設定
SSH_KEY = "~/.ssh/lambda_cloud_key"
LAMBDA_IP = "150.136.47.58"
LAMBDA_USER = "ubuntu"
CODE_PATH = "/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux"
VENV_PATH = "/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv"

# Hugging Face Token設定
HF_TOKEN_FILE = "hf_token.txt"

def run_ssh_command(command, use_tmux=False, session_name=None, detach=False, timeout=None):
    """SSH経由でコマンドを実行"""
    # TensorFlow初期化を無効化する環境変数を追加
    tf_disable_env = "export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && "
    ssh_base = f"ssh -i {SSH_KEY} {LAMBDA_USER}@{LAMBDA_IP}"
    
    if use_tmux:
        session_name = session_name or "lisa_dev"
        if detach:
            # バックグラウンドで tmux セッションを作成し、コマンドを実行
            tmux_cmd = f"tmux new-session -d -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}'"
        else:
            # tmux セッションにアタッチ
            tmux_cmd = f"tmux attach-session -t {session_name} || tmux new-session -s {session_name} 'cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}'"
        
        full_command = f"{ssh_base} \"{tmux_cmd}\""
    else:
        full_command = f"{ssh_base} \"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && {tf_disable_env}{command}\""
    
    print(f"🚀 実行中: {full_command}")
    
    if detach:
        # バックグラウンド実行
        process = subprocess.Popen(full_command, shell=True)
        print(f"✅ バックグラウンドで実行開始 (PID: {process.pid})")
        return process
    else:
        # 前景実行
        try:
            if timeout:
                result = subprocess.run(full_command, shell=True, capture_output=True, text=True, timeout=timeout)
            else:
                result = subprocess.run(full_command, shell=True, capture_output=True, text=True)
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)

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

def setup_instance():
    """新しいLambda Cloudインスタンスをセットアップ"""
    print("🏗️  Lambda Cloudインスタンスをセットアップ中...")
    
    setup_commands = [
        # システム更新と基本ツールインストール
        "sudo apt-get update",
        "sudo apt-get install -y tmux htop tree",
        
        # シンボリックリンク作成
        "ln -sfn /lambda/nfs/lisa-gemma-project-fs ~/persistent_storage",
        "ln -sfn /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux ~/project",
        
        # 環境確認
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python config_lambda_cloud.py"
    ]
    
    print("📋 実行するセットアップ手順:")
    for i, cmd in enumerate(setup_commands, 1):
        print(f"  {i}. {cmd}")
    
    response = input("\n⚠️  セットアップを実行しますか？ (y/N): ")
    if response.lower() != 'y':
        print("❌ セットアップをキャンセルしました")
        return
    
    # セットアップ実行
    for i, cmd in enumerate(setup_commands, 1):
        print(f"\n🔧 ステップ {i}/{len(setup_commands)}: {cmd}")
        result = run_ssh_command(cmd, use_tmux=False)
        if result.returncode != 0:
            print(f"❌ ステップ {i} でエラーが発生しました")
            return False
    
    print("\n✅ セットアップ完了！")
    print("📂 利用可能なショートカット:")
    print("  ~/persistent_storage → Persistent Filesystem")
    print("  ~/project → プロジェクトルート")
    
    return True

def check_cloud():
    """Lambda Cloud環境をチェック"""
    print("🔍 Lambda Cloud環境をチェック中...")
    
    checks = {}
    
    # SSH接続をチェック
    print("📋 SSH接続をチェック中...")
    result = run_ssh_command("echo 'SSH OK'", use_tmux=False)
    checks["SSH接続"] = result.returncode == 0
    print(f"   {'✅' if checks['SSH接続'] else '❌'} SSH接続")
    
    # Filesystemをチェック
    print("📋 Filesystem存在をチェック中...")
    result = run_ssh_command("ls -la /lambda/nfs/lisa-gemma-project-fs", use_tmux=False)
    checks["Filesystem存在"] = result.returncode == 0
    print(f"   {'✅' if checks['Filesystem存在'] else '❌'} Filesystem存在")
    
    # 仮想環境をチェック
    print("📋 仮想環境をチェック中...")
    result = run_ssh_command(f"source {VENV_PATH}/bin/activate && python --version", use_tmux=False)
    checks["仮想環境"] = result.returncode == 0
    print(f"   {'✅' if checks['仮想環境'] else '❌'} 仮想環境")
    
    # GPU認識をチェック
    print("📋 GPU認識をチェック中...")
    result = run_ssh_command("nvidia-smi --query-gpu=name --format=csv,noheader", use_tmux=False)
    checks["GPU認識"] = result.returncode == 0
    print(f"   {'✅' if checks['GPU認識'] else '❌'} GPU認識")
    
    # プロジェクト設定をチェック
    print("📋 プロジェクト設定をチェック中...")
    result = run_ssh_command(f"cd {CODE_PATH} && python config_lambda_cloud.py", use_tmux=False)
    checks["プロジェクト設定"] = result.returncode == 0
    print(f"   {'✅' if checks['プロジェクト設定'] else '❌'} プロジェクト設定")
    
    # パッケージをチェック
    print("📋 必要パッケージをチェック中...")
    result = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && pip list | grep -E 'torch|transformers|deepspeed'",
        use_tmux=False
    )
    # 出力から必要なパッケージが見つかるかチェック
    checks["必要パッケージ"] = result.returncode == 0
    print(f"   {'✅' if checks['必要パッケージ'] else '❌'} 必要パッケージ")
    
    # Hugging Face認証をチェック
    print("📋 Hugging Face認証をチェック中...")
    result = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && huggingface-cli whoami",
        use_tmux=False
    )
    checks["HF認証"] = result.returncode == 0
    print(f"   {'✅' if checks['HF認証'] else '❌'} Hugging Face認証")
    
    print("\n📊 環境チェック結果:")
    print("=" * 30)
    for check_name, status in checks.items():
        print(f"{'✅' if status else '❌'} {check_name}")
    
    all_passed = all(checks.values())
    if all_passed:
        print("\n🎉 全てのチェックが成功しました！")
    else:
        print("\n⚠️  一部のチェックが失敗しています。修正が必要です。")
        if not checks.get("HF認証", False):
            print("💡 Hugging Face認証が必要です: python lambda_dev_utils.py setup_hf <your_token>")
        
    return all_passed

def install_missing_packages():
    """不足パッケージの追加インストール"""
    print("📦 追加パッケージをインストール中...")
    
    additional_packages = [
        "mlflow",
        "boto3", 
        "tensorboard",
        "wandb",
        "matplotlib",
        "seaborn"
    ]
    
    install_cmd = f"source {VENV_PATH}/bin/activate && pip install " + " ".join(additional_packages)
    
    print(f"インストール対象: {', '.join(additional_packages)}")
    response = input("インストールを実行しますか？ (y/N): ")
    
    if response.lower() == 'y':
        result = run_ssh_command(install_cmd, use_tmux=False)
        if result.returncode == 0:
            print("✅ パッケージインストール完了")
        else:
            print("❌ パッケージインストール失敗")
        return result.returncode == 0
    else:
        print("❌ インストールをキャンセルしました")
        return False

def run_validation_test():
    """エンドツーエンド検証テスト"""
    print("🧪 エンドツーエンド検証テストを実行中...")
    
    # まずコードを同期
    if not sync_code():
        print("❌ コード同期に失敗しました")
        return False
    
    # 検証スクリプトを実行
    test_commands = [
        # GPU確認
        "nvidia-smi",
        
        # 基本的なPythonテスト
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python -c \"import torch; print(f'PyTorch: {{torch.__version__}}'); print(f'CUDA available: {{torch.cuda.is_available()}}'); print(f'GPU count: {{torch.cuda.device_count()}}')\"",
        
        # 設定ファイルテスト
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python config_lambda_cloud.py",
        
        # 簡単なモデルテスト（利用可能な場合）
        f"cd {CODE_PATH} && source {VENV_PATH}/bin/activate && python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/gemma-2-9b-it'); print('Tokenizer loaded successfully')\"" if os.path.exists("test_basic_model.py") else "echo 'Model test skipped'"
    ]
    
    print("🧪 実行する検証テスト:")
    for i, cmd in enumerate(test_commands, 1):
        print(f"  {i}. {cmd[:60]}...")
    
    success_count = 0
    for i, cmd in enumerate(test_commands, 1):
        print(f"\n🧪 テスト {i}/{len(test_commands)}")
        result = run_ssh_command(cmd, use_tmux=False)
        if result.returncode == 0:
            print(f"✅ テスト {i} 成功")
            success_count += 1
        else:
            print(f"❌ テスト {i} 失敗")
    
    print(f"\n📊 検証結果: {success_count}/{len(test_commands)} テスト成功")
    
    if success_count == len(test_commands):
        print("🎉 全ての検証テストが成功しました！")
        print("🚀 Lambda Cloud環境は使用準備完了です")
        return True
    else:
        print("⚠️  一部のテストが失敗しています。環境の確認が必要です。")
        return False

def setup_hf_token(token=None):
    """Hugging Face Tokenを安全に設定"""
    if token:
        # ローカルにトークンファイルを作成（.gitignoreで除外済み）
        with open(HF_TOKEN_FILE, 'w') as f:
            f.write(token.strip())
        print(f"✅ Hugging Face Tokenをローカルに保存しました: {HF_TOKEN_FILE}")
        
        # Lambda CloudでHugging Face CLIに直接ログイン
        print("🔐 Lambda CloudでHugging Face CLIにログイン中...")
        result = run_ssh_command(
            f"source {VENV_PATH}/bin/activate && huggingface-cli login --token {token.strip()}",
            use_tmux=False
        )
        if result.returncode == 0:
            print("✅ Hugging Face CLIログイン成功")
            return True
        else:
            print("❌ Hugging Face CLIログイン失敗")
            print(f"エラー出力: {result.stderr}")
            print(f"標準出力: {result.stdout}")
            return False
    else:
        # 既存のトークンファイルから読み込み
        if os.path.exists(HF_TOKEN_FILE):
            with open(HF_TOKEN_FILE, 'r') as f:
                token = f.read().strip()
            return setup_hf_token(token)
        else:
            print(f"❌ Tokenファイルが見つかりません: {HF_TOKEN_FILE}")
            print("使用方法: python lambda_dev_utils.py setup_hf <your_token>")
            return False

def check_hf_auth():
    """Hugging Face認証状態をチェック"""
    print("🔐 Hugging Face認証状態をチェック中...")
    result = run_ssh_command(
        f"source {VENV_PATH}/bin/activate && huggingface-cli whoami",
        use_tmux=False
    )
    if result.returncode == 0:
        print("✅ Hugging Face認証済み")
        return True
    else:
        print("❌ Hugging Face未認証")
        return False

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("🚀 Lambda Cloud開発ユーティリティ")
        print("=" * 50)
        print("使用可能なコマンド:")
        print("  sync        - ローカルコードをLambda Cloudに同期")
        print("  train       - 学習スクリプトを実行 (tmux)")
        print("  monitor     - GPU使用状況とトレーニング状況を監視")
        print("  results     - 学習結果を取得")
        print("  setup       - 新しいインスタンスのセットアップ")
        print("  check       - 環境の健全性チェック")
        print("  install     - 追加パッケージのインストール")
        print("  validate    - エンドツーエンド検証テスト")
        print("  emergency   - 緊急停止（全tmuxセッション終了）")
        print("  setup_hf    - Hugging Face Tokenを設定")
        print("")
        print("例:")
        print("  python lambda_dev_utils.py sync")
        print("  python lambda_dev_utils.py train train_ds.py")
        print("  python lambda_dev_utils.py setup_hf hf_xxxxxxx")
        print("  python lambda_dev_utils.py check")
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
    
    elif command == "setup":
        setup_instance()
    
    elif command == "check":
        check_cloud()
    
    elif command == "install":
        install_missing_packages()
    
    elif command == "validate":
        run_validation_test()
    
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
    
    elif command == "setup_hf":
        if len(sys.argv) < 3:
            print("❌ Hugging Face Tokenを指定してください")
            return
        token = sys.argv[2]
        setup_hf_token(token)
    
    else:
        print(f"❌ 不明なコマンド: {command}")

if __name__ == "__main__":
    main() 