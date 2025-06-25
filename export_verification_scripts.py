#!/usr/bin/env python3
"""
LISA-Gemma仕様書第1-5節 検証スクリプト集出力ツール

仕様書第1節から第5節で作成した検証スクリプトとその依存関係にある
すべてのファイル（./utils/, ./model/内のスクリプト含む）を
パス付きでまとめてTXTファイルに出力します。

対象スクリプト:
- 第1節: verify_dataset_integrity.py
- 第2節: verify_input_formatting.py  
- 第3節: verify_model_architecture.py
- 第4節: verify_loss_and_gradients.py
- 第5節: overfit_single_batch.py, test_inference_pipeline.py, verify_config_and_setup.py
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from datetime import datetime

class VerificationScriptsExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"verification_scripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.processed_files = set()
        self.related_files = []
        
        # 仕様書第1-5節の検証スクリプト
        self.verification_scripts = [
            # 第1節: データセット構築と完全性の検証
            "verify_dataset_integrity.py",
            
            # 第2節: Gemma向けマルチモーダル入力フォーマットの検証
            "verify_input_formatting.py",
            
            # 第3節: 視覚-言語アライメントのためのアーキテクチャ監査
            "verify_model_architecture.py",
            
            # 第4節: 損失計算と勾配伝播の精査
            "verify_loss_and_gradients.py",
            
            # 第5節: 堅牢性と将来の開発に向けた事前検証
            "overfit_single_batch.py",
            "test_inference_pipeline.py", 
            "verify_config_and_setup.py",
        ]
        
        # 重要な関連ファイル（手動追加）
        self.important_files = [
            "config_linux.py",
            
            # utilsディレクトリの主要ファイル（優先度高）
            "utils/dataset.py",
            "utils/data_processing.py",
            "utils/conversation.py",
            "utils/constants.py",
            
            # modelディレクトリの主要ファイル（優先度高）
            "model/LISA.py",
            "model/gemma_lisa.py", 
            "model/losses.py",
        ]
        
    def find_imports_in_file(self, file_path):
        """ファイル内のimport文を解析して依存関係を取得"""
        imports = set()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)
        except Exception as e:
            print(f"警告: {file_path} の解析に失敗: {e}")
        
        return imports
    
    def resolve_import_to_file(self, import_name):
        """import名を実際のファイルパスに解決（優先度フィルタリング）"""
        possible_files = []
        
        # プロジェクト内のローカルモジュールを検索
        parts = import_name.split('.')
        
        # 優先度の低いモジュールをスキップ
        low_priority_modules = {
            'utils.sem_seg_dataset', 'utils.refer_seg_dataset', 'utils.vqa_dataset', 
            'utils.reason_seg_dataset', 'utils.refer', 'utils.grefer',
            'model.segment_anything.utils', 'model.segment_anything.modeling'
        }
        
        if import_name in low_priority_modules:
            return possible_files
        
        # utils.dataset -> utils/dataset.py（主要ファイルのみ）
        if len(parts) >= 2 and parts[0] in ['utils', 'model']:
            potential_path = self.project_root / '/'.join(parts[:-1]) / f"{parts[-1]}.py"
            if potential_path.exists():
                # 主要ファイルのみ追加
                if self.is_priority_file(potential_path):
                    possible_files.append(potential_path)
        
        # config_linux -> config_linux.py
        if len(parts) == 1:
            potential_path = self.project_root / f"{parts[0]}.py"
            if potential_path.exists() and parts[0] == 'config_linux':
                possible_files.append(potential_path)
        
        return possible_files
    
    def is_priority_file(self, file_path):
        """ファイルが優先度の高いファイルかどうか判定"""
        priority_files = {
            'dataset.py', 'data_processing.py', 'conversation.py', 'constants.py',
            'LISA.py', 'gemma_lisa.py', 'losses.py', 'config_linux.py'
        }
        return file_path.name in priority_files
    
    def collect_related_files(self, start_file):
        """再帰的に関連ファイルを収集"""
        start_path = Path(start_file).resolve()
        
        if start_path in self.processed_files or not start_path.exists():
            return
        
        self.processed_files.add(start_path)
        self.related_files.append(start_path)
        
        print(f"処理中: {start_path.relative_to(self.project_root)}")
        
        # ファイル内のimportを解析
        imports = self.find_imports_in_file(start_path)
        
        for import_name in imports:
            # 標準ライブラリやサードパーティライブラリをスキップ
            if self.is_standard_or_third_party(import_name):
                continue
                
            # プロジェクト内のファイルを検索
            possible_files = self.resolve_import_to_file(import_name)
            for file_path in possible_files:
                self.collect_related_files(file_path)
    
    def is_standard_or_third_party(self, import_name):
        """標準ライブラリまたはサードパーティライブラリかどうか判定"""
        standard_libs = {
            'os', 'sys', 'json', 'glob', 'random', 'datetime', 'pathlib',
            'collections', 'functools', 'itertools', 'typing', 'dataclasses',
            'argparse', 'logging', 'warnings', 'traceback', 'copy', 'time',
            'yaml', 'ast', 'importlib'
        }
        
        third_party_libs = {
            'torch', 'torchvision', 'torchaudio', 'transformers', 'accelerate',
            'peft', 'bitsandbytes', 'huggingface_hub', 'deepspeed',
            'pycocotools', 'numpy', 'tqdm', 'tensorboard', 'PIL', 'Pillow',
            'cv2', 'matplotlib', 'seaborn', 'pandas', 'scipy', 'sklearn'
        }
        
        base_name = import_name.split('.')[0]
        return base_name in standard_libs or base_name in third_party_libs
    
    def add_important_files(self):
        """重要なファイルを手動で追加"""
        for file_path in self.important_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path not in self.processed_files:
                self.related_files.append(full_path)
                self.processed_files.add(full_path)
                print(f"手動追加: {file_path}")
    
    def categorize_files(self):
        """ファイルをカテゴリ別に分類"""
        categories = {
            '🔍 検証スクリプト (第1-5節)': [],
            '⚙️ 設定ファイル': [],
            '🧮 モデル実装 (コア)': [],
            '📦 データ処理・ユーティリティ (コア)': [],
        }
        
        for file_path in self.related_files:
            relative_path = file_path.relative_to(self.project_root)
            file_name = file_path.name
            
            if file_name in self.verification_scripts:
                categories['🔍 検証スクリプト (第1-5節)'].append(file_path)
            elif file_name == 'config_linux.py':
                categories['⚙️ 設定ファイル'].append(file_path)
            elif str(relative_path).startswith('model/') and file_name in ['LISA.py', 'gemma_lisa.py', 'losses.py']:
                categories['🧮 モデル実装 (コア)'].append(file_path)
            elif str(relative_path).startswith('utils/') and file_name in ['dataset.py', 'data_processing.py', 'conversation.py', 'constants.py']:
                categories['📦 データ処理・ユーティリティ (コア)'].append(file_path)
        
        return categories
    
    def export_to_txt(self):
        """関連ファイルをTXTファイルに出力"""
        # ファイルをカテゴリ別に分類
        categories = self.categorize_files()
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            # ヘッダー
            f.write("=" * 80 + "\n")
            f.write("LISA-Gemma 仕様書第1-5節 検証スクリプト集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"プロジェクトルート: {self.project_root}\n")
            f.write(f"総ファイル数: {len(self.related_files)}\n")
            f.write("=" * 80 + "\n\n")
            
            # 仕様書セクション概要
            f.write("📚 仕様書セクション概要\n")
            f.write("-" * 40 + "\n")
            f.write("第1節: データセット構築と完全性の検証 (HybridDataset)\n")
            f.write("第2節: Gemma向けマルチモーダル入力フォーマットの検証\n")
            f.write("第3節: 視覚-言語アライメントのためのアーキテクチャ監査\n")
            f.write("第4節: 損失計算と勾配伝播の精査\n")
            f.write("第5節: 堅牢性と将来の開発に向けた事前検証\n")
            f.write("  5.1. サニティチェック1: 単一バッチへの過学習\n")
            f.write("  5.2. サニティチェック2: エンドツーエンドの推論パイプライン\n")
            f.write("  5.3. サニティチェック3: 設定ファイルの読み込みと検証\n\n")
            
            # カテゴリ別目次
            f.write("📋 カテゴリ別目次\n")
            f.write("-" * 40 + "\n")
            file_counter = 1
            for category, files in categories.items():
                if files:
                    f.write(f"\n{category} ({len(files)}ファイル)\n")
                    for file_path in sorted(files, key=lambda x: x.name):
                        relative_path = file_path.relative_to(self.project_root)
                        f.write(f"  {file_counter:2d}. {relative_path}\n")
                        file_counter += 1
            f.write("\n")
            
            # 各ファイルの内容（カテゴリ別）
            file_counter = 1
            for category, files in categories.items():
                if files:
                    f.write("=" * 80 + "\n")
                    f.write(f"{category}\n")
                    f.write("=" * 80 + "\n\n")
                    
                    for file_path in sorted(files, key=lambda x: x.name):
                        relative_path = file_path.relative_to(self.project_root)
                        
                        f.write("─" * 80 + "\n")
                        f.write(f"ファイル {file_counter:2d}: {relative_path}\n")
                        
                        # 検証スクリプトの場合は対応セクションを表示
                        if file_path.name in self.verification_scripts:
                            section_map = {
                                "verify_dataset_integrity.py": "第1節",
                                "verify_input_formatting.py": "第2節", 
                                "verify_model_architecture.py": "第3節",
                                "verify_loss_and_gradients.py": "第4節",
                                "overfit_single_batch.py": "第5.1節",
                                "test_inference_pipeline.py": "第5.2節",
                                "verify_config_and_setup.py": "第5.3節"
                            }
                            section = section_map.get(file_path.name, "不明")
                            f.write(f"カテゴリ: {category}\n")
                            f.write(f"対応仕様書: {section}\n")
                        
                        f.write(f"絶対パス: {file_path}\n")
                        f.write(f"サイズ: {file_path.stat().st_size:,} bytes\n")
                        f.write("─" * 80 + "\n")
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as source_file:
                                content = source_file.read()
                                f.write(content)
                                if not content.endswith('\n'):
                                    f.write('\n')
                        except Exception as e:
                            f.write(f"❌ ファイル読み込みエラー: {e}\n")
                        
                        f.write("\n\n")
                        file_counter += 1
        
        print(f"✅ 出力完了: {self.output_file}")
        print(f"📊 総ファイル数: {len(self.related_files)}")
        
        # ファイルサイズを表示
        output_size = Path(self.output_file).stat().st_size
        print(f"📄 出力ファイルサイズ: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
    
    def run(self):
        """メイン実行関数"""
        print("🔍 LISA-Gemma仕様書第1-5節検証スクリプト関連ファイルを収集中...")
        print(f"📁 プロジェクトルート: {self.project_root}")
        
        # 検証スクリプトから開始して関連ファイルを収集
        for script_name in self.verification_scripts:
            script_path = self.project_root / script_name
            if script_path.exists():
                print(f"\n📋 {script_name} の依存関係を解析中...")
                self.collect_related_files(script_path)
            else:
                print(f"⚠️ 警告: {script_name} が見つかりません")
        
        # 重要なファイルを手動で追加
        print(f"\n📦 重要ファイルを手動追加中...")
        self.add_important_files()
        
        # TXTファイルに出力
        print(f"\n📝 {self.output_file} に出力中...")
        self.export_to_txt()
        
        # 統計情報を表示
        self.show_statistics()
    
    def show_statistics(self):
        """統計情報を表示"""
        print("\n📊 統計情報:")
        print("-" * 40)
        
        categories = self.categorize_files()
        file_types = {}
        total_size = 0
        
        # カテゴリ別統計
        for category, files in categories.items():
            if files:
                category_size = sum(f.stat().st_size for f in files)
                print(f"  {category}: {len(files)} ファイル, {category_size:,} bytes")
        
        # 拡張子別統計
        print(f"\n📄 拡張子別統計:")
        for file_path in self.related_files:
            ext = file_path.suffix.lower()
            if ext not in file_types:
                file_types[ext] = {'count': 0, 'size': 0}
            
            file_size = file_path.stat().st_size
            file_types[ext]['count'] += 1
            file_types[ext]['size'] += file_size
            total_size += file_size
        
        for ext, info in sorted(file_types.items()):
            print(f"  {ext or '(拡張子なし)'}: {info['count']} ファイル, {info['size']:,} bytes")
        
        print(f"\n📈 合計: {len(self.related_files)} ファイル, {total_size:,} bytes")

def main():
    """メイン関数"""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "."
    
    exporter = VerificationScriptsExporter(project_root)
    exporter.run()

if __name__ == "__main__":
    main() 