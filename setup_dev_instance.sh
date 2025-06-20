#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- プロジェクト設定 ---
PROJECT_NAME="LISA-Gemma-Linux"
FILESYSTEM_NAME="lisa-gemma-project-fs"
VENV_NAME="lisa_gemma_venv"
HF_TOKEN="${HF_TOKEN:-your_token_here}"  # 環境変数から取得、未設定時はプレースホルダー
# --- 設定終了 ---

# 1. パスの定義
FS_ROOT="/lambda/nfs/${FILESYSTEM_NAME}"
PROJECT_ROOT="${FS_ROOT}/code/${PROJECT_NAME}"
VENV_PATH="${FS_ROOT}/venvs/${VENV_NAME}"

echo "=== Lambda Cloud 開発インスタンス セットアップ ==="
echo "ファイルシステムルート: ${FS_ROOT}"
echo "プロジェクトルート: ${PROJECT_ROOT}"
echo "仮想環境パス: ${VENV_PATH}"
echo ""

# Persistent Filesystemの存在確認
if [[ ! -d "${FS_ROOT}" ]]; then
    echo "❌ エラー: Persistent Filesystemがマウントされていません"
    echo "インスタンス起動時にファイルシステムがアタッチされているか確認してください"
    exit 1
fi

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "❌ エラー: プロジェクトディレクトリが見つかりません"
    echo "setup_filesystem.sh を先に実行してください"
    exit 1
fi

echo "✅ Persistent Filesystemが正常にマウントされています"
echo ""

# 2. システムの更新と基本ツールのインストール
echo "📦 システム更新と基本ツールのインストール中..."
sudo apt-get update -qq
sudo apt-get install -y tmux htop tree git-lfs curl wget

# 3. Persistent Filesystem上にPython仮想環境を作成・有効化
echo ""
echo "🐍 Python仮想環境のセットアップ中..."
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "仮想環境を ${VENV_PATH} に作成します..."
    # Lambda Stackのパッケージを継承するために --system-site-packages を使用
    python3 -m venv --system-site-packages "${VENV_PATH}"
    echo "✅ 仮想環境を作成しました"
else
    echo "✅ 既存の仮想環境を使用します"
fi

# 仮想環境を有効化
source "${VENV_PATH}/bin/activate"
echo "✅ 仮想環境が有効化されました"

# 4. プロジェクト固有の依存関係をインストール
echo ""
echo "📚 プロジェクト依存関係のインストール中..."
pip install --upgrade pip -q

# requirements.txtの存在確認とインストール
if [[ -f "${PROJECT_ROOT}/requirements.txt" ]]; then
    echo "requirements.txt からPythonの依存関係をインストールします..."
    pip install -r "${PROJECT_ROOT}/requirements.txt" -q
    echo "✅ requirements.txt の依存関係をインストールしました"
else
    echo "⚠️  requirements.txt が見つかりません。基本的なライブラリをインストールします..."
fi

# 追加ライブラリのインストール
echo "追加ライブラリをインストール中..."
pip install -q mlflow boto3 tensorboard wandb accelerate deepspeed

echo "✅ 依存関係のインストールが完了しました"

# 5. Hugging Face認証の設定
echo ""
echo "🤗 Hugging Face認証の設定中..."
if [[ -n "${HF_TOKEN}" && "${HF_TOKEN}" != "your_token_here" ]]; then
    export HF_TOKEN="${HF_TOKEN}"
    echo "${HF_TOKEN}" | huggingface-cli login --token "${HF_TOKEN}"
    echo "✅ Hugging Faceトークンが設定されました"
else
    echo "⚠️  HF_TOKEN が設定されていません。手動で設定してください"
    echo "export HF_TOKEN=your_token_here"
fi

# 6. ホームディレクトリからのアクセスを容易にするためのシンボリックリンクを作成
echo ""
echo "🔗 シンボリックリンクの作成中..."
ln -sfn "${FS_ROOT}" ~/persistent_storage
ln -sfn "${PROJECT_ROOT}" ~/project
echo "✅ シンボリックリンクを作成しました"
echo "  - ~/persistent_storage -> ${FS_ROOT}"
echo "  - ~/project -> ${PROJECT_ROOT}"

# 7. 環境変数の設定
echo ""
echo "🌍 環境変数の設定中..."

# .bashrc に環境変数を追加
cat >> ~/.bashrc << 'EOF'

# Lambda Cloud LISA-Gemma Project Settings
export LISA_PROJECT_ROOT="/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux"
export LISA_VENV_PATH="/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv"
export LISA_ARTIFACTS_PATH="/lambda/nfs/lisa-gemma-project-fs/artifacts"

# Auto-activate virtual environment when logging in
if [[ -f "$LISA_VENV_PATH/bin/activate" ]]; then
    source "$LISA_VENV_PATH/bin/activate"
fi

# Add project to Python path
export PYTHONPATH="$LISA_PROJECT_ROOT:$PYTHONPATH"

# CUDA settings
export CUDA_VISIBLE_DEVICES=0
EOF

echo "✅ 環境変数を設定しました"

# 8. GPU環境の確認
echo ""
echo "🔍 GPU環境の確認中..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "⚠️  GPU情報の取得に失敗"

# PyTorchのCUDA確認
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
" 2>/dev/null || echo "⚠️  PyTorch CUDA確認に失敗"

# 9. セットアップ完了サマリー
echo ""
echo "🎉 セットアップ完了サマリー"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📁 プロジェクトディレクトリ: ${PROJECT_ROOT}"
echo "🐍 Python仮想環境: ${VENV_PATH}"
echo "🔗 シンボリックリンク: ~/project"
echo "🤗 Hugging Face: 認証済み"
echo "💾 永続ストレージ: ${FS_ROOT}"
echo ""
echo "✅ 開発環境のセットアップが完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. cd ~/project でプロジェクトディレクトリに移動"
echo "2. 学習スクリプトのテスト実行"
echo "3. 動作確認用コマンド実行"
echo ""
echo "🚀 推奨テストコマンド:"
echo "cd ~/project"
echo "python quick_ddp_test.py  # 動作確認"
echo "python train_simple_test.py  # 基本学習テスト"
echo ""
echo "🔄 新規ログイン時の推奨手順:"
echo "source ${VENV_PATH}/bin/activate"
echo "cd ~/project" 