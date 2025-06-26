# Lambda Cloud開発ガイド

## 🎯 開発方針

### 基本原則
- **ローカル中心**: 全ての操作をローカルから実行
- **SSH経由実行**: Lambda上への直接ログインは避ける
- **コード編集**: ローカルで編集 → rsync同期
- **環境操作**: ローカルからSSHコマンドで実行
- **接続安定性**: SSH切断による誤作動を防止

### なぜローカル中心なのか？
- ✅ SSH接続切断による学習中断を防止
- ✅ ローカル環境での快適な編集体験
- ✅ 操作ミスやセッション管理エラーの回避
- ✅ 一貫した開発環境の維持

## 🚀 クイックスタート

### 🆕 新しいAI Agentが最初に行うこと

```bash
# 1. 作業ディレクトリに移動
cd ~/LISA-Gemma-Linux

# 2. 環境の健全性チェック
python lambda_dev_utils.py check

# 3. Lambda Cloud接続テスト
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "echo 'Lambda Cloud接続OK'"

# 4. 統一設定の確認
python config_linux.py

# 5. 検証スクリプトで環境確認（推奨）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python verify_config_and_setup.py"
```

### 1. 基本セットアップ
```bash
# 環境チェック（ローカルから実行）
python lambda_dev_utils.py check

# Hugging Face Token設定（初回のみ）
python lambda_dev_utils.py setup_hf your_hf_token_here
```

### 2. 開発サイクル
```bash
# 1. ローカルでコード編集
# お好みのエディタ（VSCode、Cursor等）でコードを編集

# 2. Lambda Cloudに同期
python lambda_dev_utils.py sync

# 3. 環境検証（推奨）
python lambda_dev_utils.py validate

# 4. Lambda上で学習実行（ローカルから指示）
python lambda_dev_utils.py train your_script.py

# 5. 進捗監視（ローカルから確認）
python lambda_dev_utils.py monitor
```

## 📋 コマンド一覧

| コマンド | 実行場所 | 説明 | 例 |
|---------|---------|------|-----|
| `check` | ローカル | 環境の健全性チェック | `python lambda_dev_utils.py check` |
| `setup_hf` | ローカル | Hugging Face Token設定 | `python lambda_dev_utils.py setup_hf hf_xxx` |
| `sync` | ローカル | コード同期 | `python lambda_dev_utils.py sync` |
| `validate` | ローカル | 検証スクリプト実行 | `python lambda_dev_utils.py validate` |
| `train` | ローカル | 学習実行（tmux） | `python lambda_dev_utils.py train train_ds.py` |
| `monitor` | ローカル | GPU・学習状況監視 | `python lambda_dev_utils.py monitor` |
| `results` | ローカル | 結果取得 | `python lambda_dev_utils.py results` |
| `emergency` | ローカル | 緊急停止 | `python lambda_dev_utils.py emergency` |

**重要**: 全てのコマンドはローカルディレクトリから実行してください

## 🔧 環境設定

### Lambda Cloud設定
- **インスタンス**: 150.136.47.58 (1x A10 GPU, 24GB VRAM)
- **SSH鍵**: `~/.ssh/lambda_cloud_key`
- **Persistent Filesystem**: `/lambda/nfs/lisa-gemma-project-fs/`
- **Python環境**: `lisa_gemma_venv` (最適化済み)

### ディレクトリ構造
```
/lambda/nfs/lisa-gemma-project-fs/
├── data/           # データセット（全8種類）
│   ├── dataset/    # LISA学習データ
│   └── weights/    # SAM重み (sam_vit_h_4b8939.pth)
├── code/           # プロジェクトコード（rsyncで同期）
│   └── LISA-Gemma-Linux/  # メインプロジェクト
├── artifacts/      # 学習結果
│   ├── checkpoints/
│   ├── logs/
│   └── final_models/
└── venvs/          # Python仮想環境
    └── lisa_gemma_venv/  # 最適化済み環境
```

### 統一設定管理
- **設定ファイル**: `config_linux.py` (全スクリプト共通)
- **Lambda Cloud最適化**: A10 24GB制約対応済み
- **TensorFlow回避**: パフォーマンス最適化適用済み

## 🔍 検証スクリプト

### 5つの最適化済み検証スクリプト
新しい環境や重要な変更後に実行推奨：

1. **`verify_config_and_setup.py`**: 設定・環境の完全性チェック
   - 43項目の設定検証
   - Lambda Cloud環境の健全性確認
   - 実行時間: ~30秒

2. **`verify_dataset_integrity.py`**: データセットの完全性検証
   - 全8データセットの存在確認
   - SAM重みファイルの検証
   - 実行時間: ~60秒

3. **`verify_input_formatting.py`**: 入力データ形式の検証
   - 全12データセットの形式チェック
   - トークン化・ラベル処理の確認
   - 実行時間: ~4分

4. **`verify_model_architecture.py`**: モデル構造の検証
   - LISA-Gemmaアーキテクチャの完全性
   - パラメータ数・学習率の確認
   - 実行時間: ~2分

5. **`verify_loss_and_gradients.py`**: 損失・勾配の検証
   - フォワード・バックワードパスの確認
   - A10メモリ最適化対応済み
   - 実行時間: ~3分

### 実行方法
```bash
# 個別実行（ローカルから）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && source ../../venvs/lisa_gemma_venv/bin/activate && python verify_config_and_setup.py"

# 全検証実行（将来的に lambda_dev_utils.py に統合予定）
# python lambda_dev_utils.py validate
```

## 🔐 セキュリティ

### Hugging Face Token
- **ローカル保存**: `hf_token.txt`（.gitignoreで除外済み）
- **自動設定**: Lambda Cloud上に安全に設定
- **確認**: `python lambda_dev_utils.py check`でHF認証状態を確認

### 注意事項
- Tokenファイルは絶対にGitにコミットしない
- `.gitignore`に機密ファイルパターンを追加済み

## 🎯 実践的開発ワークフロー

### Phase 1: 初期環境確認
```bash
# ローカルディレクトリで実行
cd ~/LISA-Gemma-Linux

# 1. 全体チェック
python lambda_dev_utils.py check

# 2. 必要に応じてToken設定
python lambda_dev_utils.py setup_hf your_token
```

### Phase 2: コード開発・テスト
```bash
# 1. ローカルでコード編集
# お好みのエディタでファイルを編集
# 例: train_ds.py, model/gemma_lisa.py など

# 2. 変更をLambda Cloudに同期
python lambda_dev_utils.py sync

# 3. 短いテスト実行（ローカルから指示）
python lambda_dev_utils.py train test_basic_model.py

# 4. 実行状況監視（ローカルから確認）
python lambda_dev_utils.py monitor

# 5. 必要に応じて緊急停止
python lambda_dev_utils.py emergency
```

### Phase 3: 本格学習
```bash
# 1. 最新コードを同期
python lambda_dev_utils.py sync

# 2. 本格学習開始（tmuxセッションで実行）
python lambda_dev_utils.py train train_ds.py

# 3. 定期的な進捗確認
python lambda_dev_utils.py monitor

# 4. 結果取得
python lambda_dev_utils.py results
```

### Phase 4: 環境管理・調整
```bash
# パッケージ追加（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && pip install new_package"

# 設定ファイル確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "cd /lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux && python config_lambda_cloud.py"

# GPU状況確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvidia-smi"
```

## 🚨 トラブルシューティング

### よくある問題

#### SSH接続エラー
```bash
# SSH鍵のパーミッション確認（ローカルで実行）
chmod 600 ~/.ssh/lambda_cloud_key

# 接続テスト（ローカルで実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "echo 'Connection OK'"
```

#### コード同期失敗
```bash
# 手動同期（ローカルで実行）
rsync -avz --progress --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='lambda_results' ./ ubuntu@150.136.47.58:/lambda/nfs/lisa-gemma-project-fs/code/LISA-Gemma-Linux/
```

#### 学習が停止・異常終了
```bash
# tmuxセッション確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux list-sessions"

# 緊急停止（ローカルから実行）
python lambda_dev_utils.py emergency

# プロセス確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ps aux | grep python"
```

#### 環境・パッケージ問題
```bash
# 仮想環境確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && python --version"

# パッケージ状況確認（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && pip list | grep torch"

# NumPy互換性修正（ローカルから実行）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "source /lambda/nfs/lisa-gemma-project-fs/venvs/lisa_gemma_venv/bin/activate && pip install 'numpy<2.0'"
```

## 💡 ベストプラクティス

### 開発効率化
1. **ローカル編集優先**: VSCode、Cursor等でローカル編集
2. **頻繁な同期**: 小さな変更でも`sync`コマンドで同期
3. **段階的テスト**: 短いテスト → 長時間学習の順で実行
4. **定期的監視**: `monitor`コマンドで進捗確認

### SSH接続管理
1. **直接ログイン禁止**: Lambda上への直接SSH接続は避ける
2. **ワンライナー実行**: 必要な操作は`ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "command"`形式
3. **tmux活用**: 長時間処理は必ずtmuxセッション経由
4. **緊急停止準備**: 問題発生時は即座に`emergency`コマンド

### コード管理
1. **ローカルGit**: ローカルでバージョン管理
2. **同期前確認**: 重要な変更前は必ずバックアップ
3. **結果取得**: 定期的に`results`コマンドで成果物を取得

### コスト最適化
1. **効率的デバッグ**: ローカルで可能な限りテスト
2. **適切なインスタンス選択**: 開発はA10、本番はH100
3. **使用後停止**: 不要時は速やかにインスタンス停止

### A10メモリ制約対応
Lambda Cloud A10 (24GB VRAM) での重要な制約と対処法：

1. **バッチサイズ調整**: 通常2 → A10では1に自動調整
2. **勾配累積**: `GRADIENT_ACCUMULATION_STEPS=8`で実効バッチサイズ維持
3. **混合精度**: `MIXED_PRECISION=True`でBF16使用
4. **勾配チェックポイント**: `GRADIENT_CHECKPOINTING=True`でメモリ削減
5. **メモリ監視**: 検証スクリプトで使用量を自動確認

```python
# config_linux.py での A10 最適化設定
BATCH_SIZE_PER_GPU = 2  # A10では自動的に1に調整
GRADIENT_ACCUMULATION_STEPS = 8
MIXED_PRECISION = True
GRADIENT_CHECKPOINTING = True
```

## 🔄 日常的な開発パターン

### 朝の作業開始
```bash
# 1. 環境確認
python lambda_dev_utils.py check

# 2. 最新状況確認
python lambda_dev_utils.py monitor

# 3. 前日の結果取得
python lambda_dev_utils.py results
```

### コード変更時
```bash
# 1. ローカルでファイル編集
# 2. 同期
python lambda_dev_utils.py sync

# 3. テスト実行
python lambda_dev_utils.py train test_script.py

# 4. 状況確認
python lambda_dev_utils.py monitor
```

### 学習実行時
```bash
# 1. 最新コード同期
python lambda_dev_utils.py sync

# 2. 学習開始
python lambda_dev_utils.py train train_ds.py

# 3. 定期確認（別ターミナルで）
watch -n 300 "python lambda_dev_utils.py monitor"
```

### 作業終了時
```bash
# 1. 結果取得
python lambda_dev_utils.py results

# 2. ローカルでGitコミット
git add .
git commit -m "作業内容の説明"

# 3. 必要に応じてインスタンス停止
# Lambda Cloudダッシュボードから手動停止
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
# tmuxセッション一覧確認（ローカルから）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux list-sessions"

# 特定セッションのログ確認（ローカルから）
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "tmux capture-pane -t training -p"
```

### 緊急時対応
```bash
# 即座の全停止
python lambda_dev_utils.py emergency

# 状況確認
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "ps aux | grep python"
ssh -i ~/.ssh/lambda_cloud_key ubuntu@150.136.47.58 "nvidia-smi"
```

---

## 🔑 新しいAI Agentへの重要な注意事項

### 必須の理解事項
1. **全ての操作はローカルから実行**: Lambda上への直接SSH接続は禁止
2. **統一設定管理**: `config_linux.py`が全スクリプトの設定源
3. **A10メモリ制約**: 24GB制約があり、自動最適化が適用済み
4. **検証スクリプト活用**: 環境変更後は必ず検証を実行
5. **TensorFlow回避**: パフォーマンス最適化のため、TF関連は遅延読み込み

### 開発の基本フロー
```
ローカル編集 → sync → 検証 → train → monitor → results
```

### 緊急時の対応
```bash
# 即座の全停止
python lambda_dev_utils.py emergency

# 環境確認
python lambda_dev_utils.py check
```

---

**このガイドは`lambda_dev_utils.py`の機能と最適化済み検証スクリプトに基づいています。最新の機能については`python lambda_dev_utils.py`でヘルプを確認してください。**

**重要**: 全ての操作はローカルディレクトリから実行し、Lambda上への直接ログインは避けてください。 