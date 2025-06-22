#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- Lambda Cloud設定 ---
INSTANCE_IP="132.145.192.105"
REMOTE_USER="ubuntu"
SSH_KEY_PATH="~/.ssh/lambda_cloud_key"  # SSH鍵のパス
FILESYSTEM_ROOT="/lambda/nfs/lisa-gemma-project-fs"

# --- ローカル設定 ---
LOCAL_DATASET_BASE="/mnt/h/download/LISA-dataset/dataset"
LOCAL_PASCAL_PART="${LOCAL_DATASET_BASE}/vlpart/pascal_part"
LOCAL_PACO="${LOCAL_DATASET_BASE}/vlpart/paco"

# --- リモート設定 ---
REMOTE_DATASET_BASE="${FILESYSTEM_ROOT}/data/dataset"
REMOTE_PASCAL_PART="${REMOTE_DATASET_BASE}/vlpart/pascal_part"
REMOTE_PACO="${REMOTE_DATASET_BASE}/vlpart/paco"

echo "=== Lambda Cloud データ転送スクリプト ==="
echo "インスタンス: ${INSTANCE_IP}"
echo "ファイルシステム: ${FILESYSTEM_ROOT}"
echo ""

# SSH接続テスト
echo "1. SSH接続テスト..."
if ssh -i "${SSH_KEY_PATH}" -o ConnectTimeout=10 "${REMOTE_USER}@${INSTANCE_IP}" "echo 'SSH接続成功'"; then
    echo "✅ SSH接続成功"
else
    echo "❌ SSH接続失敗 - SSH鍵のパスを確認してください: ${SSH_KEY_PATH}"
    exit 1
fi

# リモートディレクトリ構造確認・作成
echo ""
echo "2. リモートディレクトリ構造確認・作成..."
ssh -i "${SSH_KEY_PATH}" "${REMOTE_USER}@${INSTANCE_IP}" "
    echo 'ファイルシステム確認:'
    ls -la /lambda/nfs/ || echo 'ファイルシステムが見つかりません'
    
    echo 'ディレクトリ作成:'
    mkdir -p ${REMOTE_DATASET_BASE}/vlpart
    echo '✅ ディレクトリ作成完了'
"

# 関数定義
transfer_pascal_part_to_remote() {
    echo ""
    echo "3. Pascal Part: ローカル → リモート転送..."
    echo "転送元: ${LOCAL_PASCAL_PART}"
    echo "転送先: ${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_PASCAL_PART}"
    
    if [ ! -d "${LOCAL_PASCAL_PART}" ]; then
        echo "❌ ローカルのPascal Partディレクトリが見つかりません: ${LOCAL_PASCAL_PART}"
        return 1
    fi
    
    # 既存のリモートディレクトリをバックアップ
    ssh -i "${SSH_KEY_PATH}" "${REMOTE_USER}@${INSTANCE_IP}" "
        if [ -d '${REMOTE_PASCAL_PART}' ]; then
            echo 'リモートの既存Pascal Partをバックアップ中...'
            mv '${REMOTE_PASCAL_PART}' '${REMOTE_PASCAL_PART}.backup.$(date +%Y%m%d_%H%M%S)'
        fi
    "
    
    # rsyncで転送
    rsync -avz --info=progress2 \
        -e "ssh -i ${SSH_KEY_PATH}" \
        "${LOCAL_PASCAL_PART}/" \
        "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_PASCAL_PART}/"
    
    echo "✅ Pascal Part転送完了"
}

transfer_paco_to_local() {
    echo ""
    echo "4. PACO: リモート → ローカル転送..."
    echo "転送元: ${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_PACO}"
    echo "転送先: ${LOCAL_PACO}"
    
    # リモートディレクトリ存在確認
    if ! ssh -i "${SSH_KEY_PATH}" "${REMOTE_USER}@${INSTANCE_IP}" "[ -d '${REMOTE_PACO}' ]"; then
        echo "❌ リモートのPACOディレクトリが見つかりません: ${REMOTE_PACO}"
        return 1
    fi
    
    # ローカルディレクトリ作成
    mkdir -p "$(dirname "${LOCAL_PACO}")"
    
    # 既存のローカルディレクトリをバックアップ
    if [ -d "${LOCAL_PACO}" ]; then
        echo "ローカルの既存PACOをバックアップ中..."
        mv "${LOCAL_PACO}" "${LOCAL_PACO}.backup.$(date +%Y%m%d_%H%M%S)"
    fi
    
    # rsyncで転送
    rsync -avz --info=progress2 \
        -e "ssh -i ${SSH_KEY_PATH}" \
        "${REMOTE_USER}@${INSTANCE_IP}:${REMOTE_PACO}/" \
        "${LOCAL_PACO}/"
    
    echo "✅ PACO転送完了"
}

# 転送実行
case "${1:-both}" in
    "pascal")
        transfer_pascal_part_to_remote
        ;;
    "paco")
        transfer_paco_to_local
        ;;
    "both"|"")
        transfer_pascal_part_to_remote
        transfer_paco_to_local
        ;;
    *)
        echo "使用法: $0 [pascal|paco|both]"
        exit 1
        ;;
esac

echo ""
echo "🎉 データ転送完了！"
echo ""
echo "確認コマンド:"
echo "  リモート確認: ssh -i ${SSH_KEY_PATH} ${REMOTE_USER}@${INSTANCE_IP} 'ls -la ${REMOTE_DATASET_BASE}/vlpart/'"
echo "  ローカル確認: ls -la ${LOCAL_DATASET_BASE}/vlpart/" 