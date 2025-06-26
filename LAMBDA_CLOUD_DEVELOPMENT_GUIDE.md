# Lambda Cloud開発ガイド 🚀

## 📋 目次
- [開発方針](#開発方針)
- [環境構成](#環境構成)
- [初期セットアップ](#初期セットアップ)
- [日常開発ワークフロー](#日常開発ワークフロー)
- [学習実行手順](#学習実行手順)
- [モニタリング](#モニタリング)
- [トラブルシューティング](#トラブルシューティング)
- [緊急停止手順](#緊急停止手順)
- [ベストプラクティス](#ベストプラクティス)

---

## 🎯 開発方針

### 基本原則
- **ローカル編集**: コードはローカル環境で快適に編集
- **リモート実行**: 計算処理はLambda Cloud上で実行
- **tmux活用**: SSH切断に対する耐性を確保
- **Persistent Filesystem**: 全てのデータ・結果をクラウド上に永続保存
- **コスト効率**: 必要時のみGPUリソースを使用

### アーキテクチャ
```
ローカル環境 (編集・管理)
    ↕ rsync同期
Lambda Cloud (実行・保存)
├── Persistent Filesystem
│   ├── data/ (データセット - 変更禁止)
│   ├── code/ (同期されたコード)
│   ├── artifacts/ (学習結果)
│   └── venvs/ (Python環境)
└── GPU Instance (計算リソース)
```

---

## 🏗️ 環境構成

### ローカル環境
- **OS**: Linux (WSL2)
- **Python**: 3.10+
- **Git**: ブランチ管理
- **SSH**: Lambda Cloud接続

### Lambda Cloud環境
- **Instance**: 1x A10 GPU (開発時) / 8x H100 (本番時)
- **Filesystem**: `/lambda/nfs/lisa-gemma-project-fs/`
- **Python環境**: `lisa_gemma_venv`
- **tmux**: セッション管理

### 主要ファイル
- `lambda_dev_utils.py`: 開発ユーティリティ（メインツール）
- `code/LISA-Gemma-Linux/config_lambda_cloud.py`: Lambda Cloud専用設定

---

## 🚀 初期セットアップ

### 1. ローカル環境準備

```bash
# ブランチ確認・作成
cd ~/LISA-Gemma-Linux
git checkout specification-validation-test2
git pull origin specification-validation-test2
git checkout -b lambda_dev

# 開発ツールの実行可能化
chmod +x lambda_dev_utils.py
```

### 2. Lambda Cloud環境確認

```bash
# 環境状況確認
python lambda_dev_utils.py gpu

# 設定ファイル確認
python lambda_dev_utils.py sync
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && python config_lambda_cloud.py"
```

### 3. 初回同期

```bash
# コードをLambda Cloudに同期
python lambda_dev_utils.py sync
```

---

## 🔄 日常開発ワークフロー

### 基本サイクル

```bash
# 1. ローカルでコード編集
# お好みのエディタ（VSCode、Cursor等）でコードを編集

# 2. Lambda Cloudに同期
python lambda_dev_utils.py sync

# 3. 動作確認
python lambda_dev_utils.py gpu
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python your_test_script.py"

# 4. 変更をコミット（ローカル）
git add .
git commit -m "feat: 変更内容の説明"
```

### ファイル同期ルール

**同期対象** ✅
- `.py` ファイル（全てのPythonコード）
- `.json` ファイル（設定ファイル）
- `.txt` ファイル（requirements等）
- `.sh` ファイル（スクリプト）

**同期除外** ❌
- `.git/` （バージョン管理情報）
- `__pycache__/` （Pythonキャッシュ）
- `*.pyc` （コンパイル済みPython）
- `lambda_results/` （ローカル結果フォルダ）
- `runs/` （ローカル実行履歴）

---

## 🏋️ 学習実行手順

### A. 短時間テスト実行

```bash
# 1. コード同期
python lambda_dev_utils.py sync

# 2. 短時間テスト（前景実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python test_script.py --max_steps 10"
```

### B. 本格学習実行（推奨）

```bash
# 1. 学習開始（tmuxバックグラウンド）
python lambda_dev_utils.py train train_ds.py my_experiment_name

# または実験名自動生成
python lambda_dev_utils.py train train_ds.py

# 💡 SSH接続が切れても学習継続！
```

### C. 学習パラメータ指定

```bash
# 複雑なパラメータを指定したい場合
python lambda_dev_utils.py sync
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux new-session -d -s my_training 'cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python train_ds.py --batch_size 8 --learning_rate 1e-4 --epochs 5'"
```

---

## 📊 モニタリング

### 学習状況確認

```bash
# 総合モニタリング
python lambda_dev_utils.py monitor

# GPU使用状況のみ
python lambda_dev_utils.py gpu

# tmuxセッション一覧
python lambda_dev_utils.py tmux-list
```

### tmuxセッション操作

```bash
# 特定のセッションにアタッチ（学習ログを見る）
python lambda_dev_utils.py attach training_my_experiment

# tmuxセッション内での操作
Ctrl+B, D    # セッションから離脱（学習は継続）
Ctrl+C       # 学習を停止
exit         # セッション終了
```

### ログファイル確認

```bash
# ログファイルを直接確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tail -f /lambda/nfs/lisa-gemma-project-fs/artifacts/logs/latest.log"

# または結果をローカルに取得
python lambda_dev_utils.py results
ls lambda_results/logs/
```

---

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. SSH接続エラー
```bash
# 接続テスト
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "echo 'Connection OK'"

# 鍵権限確認
chmod 600 ~/.ssh/lambda_cloud_key
```

#### 2. コード同期失敗
```bash
# 手動同期（詳細表示）
rsync -avz --progress --exclude='.git' --exclude='__pycache__' ./ ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

#### 3. 仮想環境エラー
```bash
# 仮想環境確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && python --version"

# パッケージ再インストール
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && pip install -r /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/requirements.txt"
```

#### 4. GPU認識エラー
```bash
# GPU状況確認
python lambda_dev_utils.py gpu

# CUDA確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvcc --version"
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "python -c 'import torch; print(torch.cuda.is_available())'"
```

#### 5. メモリ不足
```bash
# プロセス確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ps aux | grep python"

# メモリ使用量確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "free -h"
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvidia-smi"
```

---

## 🚨 緊急停止手順

### 即座の全停止
```bash
# 🚨 全Pythonプロセス緊急停止
python lambda_dev_utils.py emergency
```

### 段階的停止

```bash
# 1. 特定のtmuxセッションのみ停止
python lambda_dev_utils.py kill training_my_experiment

# 2. 全tmuxセッション確認
python lambda_dev_utils.py tmux-list

# 3. 手動でプロセス確認・停止
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ps aux | grep python"
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "kill <PID>"
```

### インスタンス停止前の準備
```bash
# 1. 重要な結果をローカルに保存
python lambda_dev_utils.py results

# 2. 現在の状況をバックアップ
git add .
git commit -m "backup: インスタンス停止前のバックアップ"
git push origin lambda_dev

# 3. Lambda Cloudのダッシュボードからインスタンス停止
```

---

## 💡 ベストプラクティス

### 開発時
- **小さな変更を頻繁にテスト**: 大きな変更をいきなり長時間学習で試さない
- **実験名を明確に**: 後で結果を特定しやすい名前を使用
- **定期的な結果取得**: `python lambda_dev_utils.py results`を習慣に

### 学習時
- **tmuxを必ず使用**: 長時間学習では必須
- **段階的なスケールアップ**: 1GPU → 4GPU → 8GPUの順で検証
- **チェックポイント保存**: 定期的な中間保存設定

### コスト管理
- **使用後は即座に停止**: 不要な課金を避ける
- **開発時は小さなインスタンス**: A10でテスト、H100で本番
- **Persistent Filesystemを活用**: データロストを防ぐ

### バックアップ
```bash
# 定期的なローカルバックアップ
python lambda_dev_utils.py results
git add lambda_results/
git commit -m "backup: 実験結果 $(date)"
```

### セキュリティ
- **SSH鍵の適切な管理**: 権限600、バックアップ保存
- **機密データの扱い**: HuggingFace tokenなどの適切な管理

---

## 📞 サポート・リファレンス

### 主要コマンド一覧
```bash
# 基本操作
python lambda_dev_utils.py sync          # コード同期
python lambda_dev_utils.py train <script> # 学習開始
python lambda_dev_utils.py monitor       # 状況確認
python lambda_dev_utils.py results       # 結果取得

# tmux操作
python lambda_dev_utils.py tmux-list     # セッション一覧
python lambda_dev_utils.py attach <name> # セッション接続
python lambda_dev_utils.py kill <name>   # セッション終了

# 緊急時
python lambda_dev_utils.py emergency     # 緊急停止
python lambda_dev_utils.py gpu           # GPU確認
```

### 設定ファイル
- `config_lambda_cloud.py`: Lambda Cloud環境の全設定
- `lambda_dev_utils.py`: 開発ワークフローの自動化

### ディレクトリ構造
```
Lambda Cloud Filesystem: /lambda/nfs/lisa-gemma-project-fs/
├── data/              # データセット（保護）
├── code/              # 同期されたコード
├── artifacts/         # 学習結果
│   ├── checkpoints/   # モデルチェックポイント
│   ├── logs/          # 学習ログ
│   └── final_models/  # 最終モデル
└── venvs/            # Python仮想環境
```

---

## 🎉 この開発環境の利点

✅ **接続安定性**: tmuxによるSSH切断耐性  
✅ **開発効率**: ローカル編集 + ワンコマンド同期  
✅ **コスト効率**: 必要時のみGPU使用  
✅ **安全性**: 緊急停止・バックアップ機能  
✅ **スケーラビリティ**: A10開発 → H100本番の段階的移行  

Happy Coding! 🚀 