#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- ユーザー設定項目 ---
LOCAL_PROJECT_PATH="/home/oda/LISA-Gemma-Linux" # ローカルのプロジェクトディレクトリ
LOCAL_DATASET_PATH="/mnt/h/download/LISA-dataset/dataset" # LISAデータセット
LOCAL_SAM_WEIGHTS_PATH="/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth" # SAMモデル重み
LAMBDA_SSH_KEY="/home/oda/.ssh/lambda_cloud_key" # Lambda CloudのSSH秘密鍵パス
INSTANCE_IP="150.136.81.226" # A10インスタンスのIPアドレス
FILESYSTEM_NAME="lisa-gemma-project-fs"
# --- 設定項目終了 ---

REMOTE_USER="ubuntu"
REMOTE_BASE_PATH="/lambda/nfs/${FILESYSTEM_NAME}"

echo "=== Lambda Cloud データ移行スクリプト ==="
echo "プロジェクト: ${LOCAL_PROJECT_PATH}"
echo "データセット: ${LOCAL_DATASET_PATH}"
echo "SAM重み: ${LOCAL_SAM_WEIGHTS_PATH}"
echo "転送先: ${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}"

# 設定チェック
if [[ "${LAMBDA_SSH_KEY}" == "<PATH_TO_YOUR_LAMBDA_SSH_KEY>" ]]; then
    echo "❌ エラー: LAMBDA_SSH_KEY を設定してください"
    echo "Lambda CloudのSSH秘密鍵のパスを指定する必要があります"
    exit 1
fi

if [[ "${INSTANCE_IP}" == "<YOUR_INSTANCE_IP>" ]]; then
    echo "❌ エラー: INSTANCE_IP を設定してください"
    echo "一時転送用インスタンスのIPアドレスを指定する必要があります"
    exit 1
fi

if [[ ! -f "${LAMBDA_SSH_KEY}" ]]; then
    echo "❌ エラー: SSH秘密鍵が見つかりません: ${LAMBDA_SSH_KEY}"
    exit 1
fi

# パス存在確認
if [[ ! -d "${LOCAL_PROJECT_PATH}" ]]; then
    echo "❌ エラー: プロジェクトパスが見つかりません: ${LOCAL_PROJECT_PATH}"
    exit 1
fi

if [[ ! -d "${LOCAL_DATASET_PATH}" ]]; then
    echo "❌ エラー: データセットパスが見つかりません: ${LOCAL_DATASET_PATH}"
    exit 1
fi

if [[ ! -f "${LOCAL_SAM_WEIGHTS_PATH}" ]]; then
    echo "❌ エラー: SAM重みファイルが見つかりません: ${LOCAL_SAM_WEIGHTS_PATH}"
    exit 1
fi

echo ""
echo "📋 転送対象の確認:"
echo "- プロジェクト: $(find ${LOCAL_PROJECT_PATH} -name "*.py" | wc -l) 個のPythonファイル"
echo "- プロジェクトサイズ: $(du -sh ${LOCAL_PROJECT_PATH} | cut -f1)"
echo "- データセットサイズ: $(du -sh ${LOCAL_DATASET_PATH} | cut -f1)"
echo "- SAM重みサイズ: $(du -sh ${LOCAL_SAM_WEIGHTS_PATH} | cut -f1)"

echo ""
read -p "続行しますか？ [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "転送をキャンセルしました。"
    exit 0
fi

echo ""
echo "🚀 データ転送を開始します..."

# 1. プロジェクトファイルの転送
echo "📁 プロジェクトファイルを転送中..."
rsync -avz --info=progress2 \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='runs/' \
    --exclude='output/' \
    --exclude='vis_output/' \
    --exclude='.vscode/' \
    -e "ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_PROJECT_PATH}/" \
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/project/"

# 2. データセットの転送
echo "📊 データセットを転送中..."
rsync -avz --info=progress2 \
    -e "ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_DATASET_PATH}/" \
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/dataset/"

# 3. SAM重みファイルの転送
echo "🎯 SAM重みファイルを転送中..."
# SAM重みファイル用のディレクトリを作成
ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no \
    ${REMOTE_USER}@${INSTANCE_IP} \
    "mkdir -p ${REMOTE_BASE_PATH}/data/weights"

# SAM重みファイルを転送
rsync -avz --info=progress2 \
    -e "ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_SAM_WEIGHTS_PATH}" \
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/weights/"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ データ転送が完了しました！"
    echo ""
    echo "📋 次のステップ:"
    echo "1. Lambda CloudインスタンスにSSHでログイン"
    echo "2. setup_filesystem.sh を実行してプロジェクト構造を確立"
    echo "3. 転送されたデータの検証"
    echo ""
    echo "SSH接続コマンド:"
    echo "ssh -i ${LAMBDA_SSH_KEY} ${REMOTE_USER}@${INSTANCE_IP}"
else
    echo "❌ データ転送中にエラーが発生しました。"
    exit 1
fi 