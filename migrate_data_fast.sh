#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- ユーザー設定項目 ---
LOCAL_PROJECT_PATH="/home/oda/LISA-Gemma-Linux"
LOCAL_DATASET_PATH="/mnt/h/download/LISA-dataset/dataset"
LOCAL_SAM_WEIGHTS_PATH="/mnt/c/Users/oda/foodlmm-llama/weights/sam_vit_h_4b8939.pth"
LAMBDA_SSH_KEY="/home/oda/.ssh/lambda_cloud_key"
INSTANCE_IP="150.136.139.228"
FILESYSTEM_NAME="lisa-gemma-project-fs"
# --- 設定項目終了 ---

REMOTE_USER="ubuntu"
REMOTE_BASE_PATH="/lambda/nfs/${FILESYSTEM_NAME}"

echo "=== Lambda Cloud 高速データ移行 ==="
echo "🚀 サイズ計算をスキップして直接転送を開始します..."
echo ""

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
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/project/" &
PROJECT_PID=$!

# 2. データセットの転送（バックグラウンド）
echo "📊 データセットを転送中..."
rsync -avz --info=progress2 \
    -e "ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_DATASET_PATH}/" \
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/dataset/" &
DATASET_PID=$!

# 3. SAM重みファイルの転送（バックグラウンド）
echo "🎯 SAM重みファイルを転送中..."
# ディレクトリ作成
ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no \
    ${REMOTE_USER}@${INSTANCE_IP} \
    "mkdir -p ${REMOTE_BASE_PATH}/data/weights"

rsync -avz --info=progress2 \
    -e "ssh -i ${LAMBDA_SSH_KEY} -o StrictHostKeyChecking=no" \
    "${LOCAL_SAM_WEIGHTS_PATH}" \
    "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_BASE_PATH}/data/weights/" &
SAM_PID=$!

echo ""
echo "⏳ 全ての転送プロセスを実行中..."
echo "プロジェクト転送PID: $PROJECT_PID"
echo "データセット転送PID: $DATASET_PID"  
echo "SAM重み転送PID: $SAM_PID"
echo ""
echo "進行状況を確認するには:"
echo "  watch -n 5 'ps aux | grep rsync'"
echo ""

# 全プロセスの完了を待機
wait $PROJECT_PID
echo "✅ プロジェクトファイル転送完了"

wait $DATASET_PID  
echo "✅ データセット転送完了"

wait $SAM_PID
echo "✅ SAM重みファイル転送完了"

echo ""
echo "🎉 全ての転送が完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. SSH接続: ssh -i ${LAMBDA_SSH_KEY} ${REMOTE_USER}@${INSTANCE_IP}"
echo "2. setup_filesystem.sh を実行"
echo "3. setup_dev_instance.sh を実行" 