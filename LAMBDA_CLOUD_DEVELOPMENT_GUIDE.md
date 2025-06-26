# Lambda Cloud開発ガイド

## 🚀 クイックスタート

### 1. 基本セットアップ
```bash
# 環境チェック
python lambda_dev_utils.py check

# Hugging Face Token設定（初回のみ）
python lambda_dev_utils.py setup_hf your_hf_token_here
```

### 2. 開発サイクル
```bash
# コード同期
python lambda_dev_utils.py sync

# 学習実行
python lambda_dev_utils.py train your_script.py

# 監視
python lambda_dev_utils.py monitor
```

## 📋 コマンド一覧

| コマンド | 説明 | 例 |
|---------|------|-----|
| `check` | 環境の健全性チェック | `python lambda_dev_utils.py check` |
| `setup_hf` | Hugging Face Token設定 | `python lambda_dev_utils.py setup_hf hf_xxx` |
| `sync` | コード同期 | `python lambda_dev_utils.py sync` |
| `train` | 学習実行（tmux） | `python lambda_dev_utils.py train train_ds.py` |
| `monitor` | GPU・学習状況監視 | `python lambda_dev_utils.py monitor` |
| `results` | 結果取得 | `python lambda_dev_utils.py results` |
| `emergency` | 緊急停止 | `python lambda_dev_utils.py emergency` |

## 🔧 環境設定

### Lambda Cloud設定
- **インスタンス**: 150.136.47.58 (1x A10 GPU)
- **SSH鍵**: `~/.ssh/lambda_cloud_key`
- **Persistent Filesystem**: `/lambda/nfs/lisa-gemma-project-fs/`

### ディレクトリ構造
```
/lambda/nfs/lisa-gemma-project-fs/
├── data/           # データセット
├── code/           # プロジェクトコード
├── artifacts/      # 学習結果
│   ├── checkpoints/
│   ├── logs/
│   └── final_models/
└── venvs/          # Python仮想環境
```

## 🔐 セキュリティ

### Hugging Face Token
- **ローカル保存**: `hf_token.txt`（.gitignoreで除外済み）
- **自動設定**: Lambda Cloud上に安全に設定
- **確認**: `python lambda_dev_utils.py check`でHF認証状態を確認

### 注意事項
- Tokenファイルは絶対にGitにコミットしない
- `.gitignore`に機密ファイルパターンを追加済み

## 🎯 開発ワークフロー

### Phase 1: 環境確認
```bash
# 1. 全体チェック
python lambda_dev_utils.py check

# 2. 必要に応じてToken設定
python lambda_dev_utils.py setup_hf your_token
```

### Phase 2: 開発・デバッグ
```bash
# 1. コード同期
python lambda_dev_utils.py sync

# 2. 短いテスト実行
python lambda_dev_utils.py train test_basic_model.py

# 3. 監視
python lambda_dev_utils.py monitor
```

### Phase 3: 本格学習
```bash
# 1. 学習開始（tmuxセッション）
python lambda_dev_utils.py train train_ds.py

# 2. 進捗監視
python lambda_dev_utils.py monitor

# 3. 結果取得
python lambda_dev_utils.py results
```

## 🚨 トラブルシューティング

### よくある問題

#### SSH接続エラー
```bash
# SSH鍵のパーミッション確認
chmod 600 ~/.ssh/lambda_cloud_key

# 接続テスト
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58
```

#### Hugging Face認証エラー
```bash
# 認証状態確認
python lambda_dev_utils.py check

# Token再設定
python lambda_dev_utils.py setup_hf your_new_token
```

#### GPU使用率が低い
```bash
# GPU状況確認
python lambda_dev_utils.py monitor

# プロセス確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvidia-smi"
```

#### 学習が停止
```bash
# tmuxセッション確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux list-sessions"

# 緊急停止
python lambda_dev_utils.py emergency
```

### パッケージ互換性問題
```bash
# NumPy互換性修正（必要に応じて）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && pip install 'numpy<2.0'"
```

## 💡 ベストプラクティス

### 開発効率化
1. **小さなテストから始める**: `test_basic_model.py`で動作確認
2. **定期的な監視**: `monitor`コマンドで進捗確認
3. **結果の定期取得**: 重要なチェックポイントで`results`実行

### コスト最適化
1. **不要時はインスタンス停止**: 学習完了後は速やかに停止
2. **効率的なデバッグ**: ローカルで可能な限りテスト
3. **バッチサイズ調整**: GPU使用率を最大化

### データ管理
1. **Persistent Filesystem活用**: 全データをクラウド上に保存
2. **定期バックアップ**: 重要な結果はローカルにも保存
3. **バージョン管理**: Git経由でコード変更を管理

## 🔄 アップデート手順

### 新しいインスタンスでの作業
```bash
# 1. 環境チェック
python lambda_dev_utils.py check

# 2. 必要に応じてセットアップ
python lambda_dev_utils.py setup

# 3. HF Token設定
python lambda_dev_utils.py setup_hf your_token
```

### コード更新
```bash
# 1. ローカルでGit操作
git pull origin main
git checkout your_branch

# 2. Lambda Cloudに同期
python lambda_dev_utils.py sync
```

## 📞 サポート

### 環境チェック
問題が発生した場合、まず環境チェックを実行：
```bash
python lambda_dev_utils.py check
```

### ログ確認
詳細なエラー情報が必要な場合：
```bash
# tmuxセッション内のログ確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58
tmux attach-session -t training
```

---

**このガイドは`lambda_dev_utils.py`の機能に基づいています。最新の機能については`python lambda_dev_utils.py`でヘルプを確認してください。** 