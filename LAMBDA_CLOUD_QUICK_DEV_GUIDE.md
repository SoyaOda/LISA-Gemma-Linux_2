# Lambda Cloud 開発クイックガイド

## 🎯 核心の開発フロー

### 基本パターン（最重要）
```bash
# 1. ローカルでファイル編集
# お好みのエディタで編集

# 2. Lambda Cloudに転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" your_file.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 3. Lambda Cloud上で実行
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && timeout 600 python your_file.py"
```

## 🚀 GPU借用後の初期セットアップ

### 1. Lambda Cloud GPU インスタンス起動
- On-demand 1x NVIDIA A10 (24GB) を選択
- **重要**: `lisa-gemma-project-fs` Persistent Filesystemを必ずアタッチ
- IP: `150.136.47.58` (例)

### 2. SSH接続確認
```bash
# 接続テスト
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "echo 'Lambda Cloud接続OK'"
```

### 3. 環境確認
```bash
# Python環境とGPU確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && python --version && nvidia-smi"
```

## 📝 実践的な開発例

### 例1: 検証スクリプトの開発・実行
```bash
# 1. ローカルでverify_loss_and_gradients.pyを編集
# VSCode、Cursor等で編集

# 2. 転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" verify_loss_and_gradients.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 3. 実行（タイムアウト付き）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && timeout 600 python verify_loss_and_gradients.py"
```

### 例2: 学習スクリプトの開発・実行
```bash
# 1. ローカルでtrain_ds.pyを編集

# 2. 転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" train_ds.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 3. tmuxセッションで長時間実行
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && tmux new-session -d -s training 'export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && python train_ds.py'"
```

### 例3: 複数ファイル同時転送
```bash
# 複数ファイルを一度に転送
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" config_linux.py train_ds.py model/gemma_lisa.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

## 🔧 よく使うコマンド集

### ファイル転送パターン
```bash
# 個別ファイル
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" file.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# 複数ファイル
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" file1.py file2.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/

# プロジェクト全体（時間がかかる）
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" ./ ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

### 実行パターン
```bash
# 短時間実行（タイムアウト付き）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && timeout 600 python script.py"

# 長時間実行（tmux使用）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && tmux new-session -d -s training 'export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && python long_script.py'"

# tmuxセッション確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux list-sessions"

# tmuxセッション接続
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 -t "tmux attach-session -t training"
```

### 監視・確認パターン
```bash
# GPU使用状況
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvidia-smi"

# ファイル存在確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ls -la /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/your_file.py"

# プロセス確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ps aux | grep python"
```

## ⚡ 高速開発のコツ

### 1. エイリアス設定（ローカル）
```bash
# ~/.bashrcに追加
alias lc-sync='rsync -avz --exclude=".git" --exclude="__pycache__" --exclude="*.pyc" --exclude=".gitignore" -e "ssh -i ~/.ssh/lambda_cloud_key"'
alias lc-run='ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 &&"'

# 使用例
lc-sync your_file.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
lc-run timeout 600 python your_file.py
```

### 2. 開発サイクル最適化
```bash
# 1回のコマンドで転送→実行
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.gitignore' -e "ssh -i ~/.ssh/lambda_cloud_key" script.py ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/ && ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && export TF_CPP_MIN_LOG_LEVEL=3 && export TF_ENABLE_ONEDNN_OPTS=0 && python script.py"
```

## 🚨 重要な注意点

### 必須の環境変数
- `TF_CPP_MIN_LOG_LEVEL=3`: TensorFlow警告を抑制
- `TF_ENABLE_ONEDNN_OPTS=0`: TensorFlowパフォーマンス最適化

### A10 GPU制約
- VRAM: 24GB制限
- バッチサイズは自動調整される（通常2→1）
- メモリ不足時は`torch.cuda.empty_cache()`が自動実行

### ファイルパス
- プロジェクトルート: `/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/`
- Python環境: `/lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/`
- データセット: `/lambda/nfs/lisa-gemma-project-fs/data/`

---

**このガイドで95%の開発作業をカバーできます。迷ったら「編集→転送→実行」の基本パターンに戻ってください。** 