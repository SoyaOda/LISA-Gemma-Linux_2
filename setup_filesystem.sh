#!/bin/bash
set -e # エラーが発生したら即座に終了

# --- プロジェクト設定 ---
FILESYSTEM_NAME="lisa-gemma-project-fs"
PROJECT_REPO="https://github.com/SoyaOda/LISA-Gemma-Linux.git"
PROJECT_DIR_NAME="LISA-Gemma-Linux"
TARGET_BRANCH="deepspeed_migration"
# --- 設定終了 ---

# パスの定義
FILESYSTEM_ROOT="/lambda/nfs/${FILESYSTEM_NAME}"

echo "=== Lambda Cloud Persistent Filesystem セットアップ ==="
echo "ファイルシステムルート: ${FILESYSTEM_ROOT}"
echo "プロジェクトリポジトリ: ${PROJECT_REPO}"
echo "ターゲットブランチ: ${TARGET_BRANCH}"
echo ""

# ファイルシステムの存在確認
if [[ ! -d "${FILESYSTEM_ROOT}" ]]; then
    echo "❌ エラー: Persistent Filesystemがマウントされていません"
    echo "インスタンス起動時にファイルシステムがアタッチされているか確認してください"
    exit 1
fi

echo "✅ Persistent Filesystemが正常にマウントされています"
echo ""

# 1. プロジェクト用のディレクトリを作成
echo "📁 プロジェクトディレクトリ構造を作成中..."
mkdir -p "${FILESYSTEM_ROOT}/code"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/checkpoints"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/logs"
mkdir -p "${FILESYSTEM_ROOT}/artifacts/final_models"
mkdir -p "${FILESYSTEM_ROOT}/venvs" # Python仮想環境用

echo "✅ ディレクトリ構造を作成しました"

# 2. 'code'ディレクトリにリポジトリをクローン
echo ""
echo "📥 GitHubリポジトリをクローン中..."
cd "${FILESYSTEM_ROOT}/code"

if [[ -d "${PROJECT_DIR_NAME}" ]]; then
    echo "⚠️  既存のリポジトリが見つかりました。更新を実行します..."
    cd "${PROJECT_DIR_NAME}"
    git fetch origin
    git checkout "${TARGET_BRANCH}"
    git pull origin "${TARGET_BRANCH}"
    echo "✅ リポジトリを更新しました"
else
    echo "📦 新規リポジトリをクローン中..."
    git clone "${PROJECT_REPO}"
    cd "${PROJECT_DIR_NAME}"
    git checkout "${TARGET_BRANCH}"
    echo "✅ リポジトリをクローンしました"
fi

# 3. 権限の設定
echo ""
echo "🔐 ファイル権限を設定中..."
chmod -R 755 "${FILESYSTEM_ROOT}"
chmod +x "${FILESYSTEM_ROOT}/code/${PROJECT_DIR_NAME}"/*.py || true

# 4. 構造の確認とサマリー表示
echo ""
echo "📊 セットアップ完了サマリー:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "📁 作成されたディレクトリ構造:"
tree "${FILESYSTEM_ROOT}" -d -L 3 2>/dev/null || find "${FILESYSTEM_ROOT}" -type d | head -20

echo ""
echo "📦 プロジェクトファイル統計:"
cd "${FILESYSTEM_ROOT}/code/${PROJECT_DIR_NAME}"
echo "- Python files: $(find . -name "*.py" | wc -l)"
echo "- JSON configs: $(find . -name "*.json" | wc -l)"
echo "- Markdown docs: $(find . -name "*.md" | wc -l)"

echo ""
echo "🔍 Git状態確認:"
git status --porcelain | head -5
echo "Current branch: $(git branch --show-current)"
echo "Latest commit: $(git log --oneline -1)"

echo ""
echo "💾 ディスク使用量:"
echo "Total filesystem usage: $(du -sh ${FILESYSTEM_ROOT} | cut -f1)"
echo "Code directory: $(du -sh ${FILESYSTEM_ROOT}/code | cut -f1)"
echo "Data directory: $(du -sh ${FILESYSTEM_ROOT}/data 2>/dev/null | cut -f1 || echo 'N/A')"

echo ""
echo "✅ セットアップが完了しました！"
echo ""
echo "📋 次のステップ:"
echo "1. setup_dev_instance.sh を実行して開発環境をセットアップ"
echo "2. プロジェクトディレクトリに移動: cd ~/project (シンボリックリンクを作成予定)"
echo "3. 学習スクリプトのテスト実行"
echo ""
echo "🚀 推奨コマンド:"
echo "cd ${FILESYSTEM_ROOT}/code/${PROJECT_DIR_NAME}"
echo "ls -la" 