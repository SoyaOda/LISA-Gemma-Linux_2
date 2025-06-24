#!/usr/bin/env python3
"""
検証スクリプト出力ツール

verify_dataset_integrity.pyとその関連スクリプト（utils/、model/内）を
パス付きでまとめてTXTファイルに出力します。
"""

import os
import sys
from pathlib import Path
from datetime import datetime

class VerificationScriptsExporter:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root).resolve()
        self.output_file = f"verification_scripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.target_files = []
        
    def collect_target_files(self):
        """対象ファイルを収集"""
        
        # 1. メインの検証スクリプト
        main_script = self.project_root / "verify_dataset_integrity.py"
        if main_script.exists():
            self.target_files.append(main_script)
        
        # 2. utils/ディレクトリの全.pyファイル
        utils_dir = self.project_root / "utils"
        if utils_dir.exists():
            for file_path in utils_dir.glob("*.py"):
                if file_path.is_file():
                    self.target_files.append(file_path)
        
        # 3. model/ディレクトリの全.pyファイル（segment_anythingは除外）
        model_dir = self.project_root / "model"
        if model_dir.exists():
            for file_path in model_dir.glob("*.py"):
                if file_path.is_file():
                    self.target_files.append(file_path)
        
        # 4. 設定ファイル等の重要なサポートファイル
        support_files = [
            "config_linux.py",
            "config_small_test.py",
            "code_verrification_test._log.md",
            "ds_config.json",
            "requirements.txt",
        ]
        
        for file_path in support_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.target_files.append(full_path)
        
        # 5. JSONファイル（utils/内の設定ファイル）
        json_files = [
            "utils/ade20k_classes.json",
            "utils/cocostuff_classes.txt",
        ]
        
        for file_path in json_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                self.target_files.append(full_path)
        
        # ファイルをパス順にソート
        self.target_files.sort(key=lambda x: str(x))
        
    def get_file_category(self, file_path):
        """ファイルのカテゴリを判定"""
        relative_path = file_path.relative_to(self.project_root)
        
        if relative_path.name == "verify_dataset_integrity.py":
            return "🔍 メイン検証スクリプト"
        elif relative_path.parts[0] == "utils":
            return "🛠️ ユーティリティモジュール"
        elif relative_path.parts[0] == "model":
            return "🧠 モデル定義"
        elif relative_path.name.startswith("config_"):
            return "⚙️ 設定ファイル"
        elif relative_path.suffix == ".json":
            return "📄 設定・データファイル"
        elif relative_path.suffix == ".md":
            return "📝 ドキュメント"
        else:
            return "📋 サポートファイル"
    
    def export_to_txt(self):
        """対象ファイルをTXTファイルに出力"""
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LISA-Gemma 検証スクリプト集\n")
            f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"プロジェクトルート: {self.project_root}\n")
            f.write(f"総ファイル数: {len(self.target_files)}\n")
            f.write("=" * 80 + "\n\n")
            
            # カテゴリ別目次
            f.write("📋 カテゴリ別目次\n")
            f.write("-" * 40 + "\n")
            
            categories = {}
            for file_path in self.target_files:
                category = self.get_file_category(file_path)
                if category not in categories:
                    categories[category] = []
                categories[category].append(file_path)
            
            for category, files in categories.items():
                f.write(f"\n{category}\n")
                for file_path in files:
                    relative_path = file_path.relative_to(self.project_root)
                    f.write(f"  • {relative_path}\n")
            
            f.write("\n\n")
            
            # 詳細目次
            f.write("📋 詳細目次\n")
            f.write("-" * 40 + "\n")
            for i, file_path in enumerate(self.target_files, 1):
                relative_path = file_path.relative_to(self.project_root)
                category = self.get_file_category(file_path)
                f.write(f"{i:2d}. {relative_path} ({category})\n")
            f.write("\n")
            
            # 各ファイルの内容
            for i, file_path in enumerate(self.target_files, 1):
                relative_path = file_path.relative_to(self.project_root)
                category = self.get_file_category(file_path)
                
                f.write("=" * 80 + "\n")
                f.write(f"ファイル {i:2d}: {relative_path}\n")
                f.write(f"カテゴリ: {category}\n")
                f.write(f"絶対パス: {file_path}\n")
                f.write(f"サイズ: {file_path.stat().st_size:,} bytes\n")
                f.write("=" * 80 + "\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as source_file:
                        content = source_file.read()
                        f.write(content)
                        if not content.endswith('\n'):
                            f.write('\n')
                except Exception as e:
                    f.write(f"❌ ファイル読み込みエラー: {e}\n")
                
                f.write("\n\n")
        
        print(f"✅ 出力完了: {self.output_file}")
        
    def show_statistics(self):
        """統計情報を表示"""
        print("\n📊 収集統計:")
        print("-" * 40)
        
        categories = {}
        total_size = 0
        
        for file_path in self.target_files:
            category = self.get_file_category(file_path)
            if category not in categories:
                categories[category] = {'count': 0, 'size': 0}
            
            file_size = file_path.stat().st_size
            categories[category]['count'] += 1
            categories[category]['size'] += file_size
            total_size += file_size
        
        for category, stats in categories.items():
            print(f"{category}: {stats['count']}ファイル ({stats['size']:,} bytes)")
        
        print(f"\n📄 総合計: {len(self.target_files)}ファイル ({total_size:,} bytes)")
        
        # 出力ファイルサイズ
        if os.path.exists(self.output_file):
            output_size = Path(self.output_file).stat().st_size
            print(f"📄 出力ファイルサイズ: {output_size:,} bytes ({output_size/1024/1024:.2f} MB)")
    
    def run(self):
        """メイン実行関数"""
        print("🔍 検証スクリプト関連ファイルを収集中...")
        print(f"📁 プロジェクトルート: {self.project_root}")
        
        # 対象ファイルを収集
        self.collect_target_files()
        
        if not self.target_files:
            print("❌ エラー: 対象ファイルが見つかりません")
            return
        
        print(f"📋 収集完了: {len(self.target_files)}ファイル")
        
        # TXTファイルに出力
        self.export_to_txt()
        
        # 統計情報を表示
        self.show_statistics()
        
        # ファイル一覧の表示
        print("\n📋 収集されたファイル一覧:")
        print("-" * 40)
        for file_path in self.target_files:
            relative_path = file_path.relative_to(self.project_root)
            category = self.get_file_category(file_path)
            print(f"  {category} {relative_path}")

def main():
    """メイン関数"""
    print("LISA-Gemma 検証スクリプト エクスポートツール")
    print("=" * 50)
    
    exporter = VerificationScriptsExporter()
    exporter.run()
    
    print("\n✅ エクスポート処理が完了しました。")

if __name__ == "__main__":
    main() 