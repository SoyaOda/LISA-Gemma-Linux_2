"""
LISA-Llama4プロジェクト用の共通定数定義
Llama-4-Scout-17B-16E-Instruct + SAM統合モデル用
"""

# デフォルトトークン
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_SEG_TOKEN = "[SEG]"

# Llama-4のチャットテンプレート用トークン (transformers標準)
LLAMA4_START_OF_TURN = "<|start_header_id|>"
LLAMA4_END_OF_TURN = "<|end_header_id|>"
LLAMA4_BOS_TOKEN = "<|begin_of_text|>"
LLAMA4_EOS_TOKEN = "<|end_of_text|>"

# 無視すべきインデックス
IGNORE_INDEX = -100

# ✅ Llama-4-Scout-17B-16E-Instruct仕様準拠設計：
# Llama-4はネイティブマルチモーダル（Early Fusion）のため、
# SEGトークン1個のみを追加し、語彙サイズは最小限に抑える。
SEG_TOKEN = "[SEG]"  # オリジナルLISAと同じ形式

# システムプロンプト
SYSTEM_PROMPT = """You are a helpful assistant that can analyze images and understand visual content. You can describe what you see in images and answer questions about them."""

LLAMA4_SYSTEM_PROMPT = """You are a helpful assistant that can analyze images and perform segmentation tasks. When asked to segment objects, you should respond with the [SEG] token."""

# Llama-4専用システムプロンプト
LLAMA4_SEGMENTATION_PROMPT = """You are LISA (Large-language Instructed Segmentation Assistant), a multimodal AI assistant powered by Llama-4-Scout-17B-16E-Instruct that can understand images and perform precise object segmentation. When asked to segment objects or regions in images, respond with the [SEG] token to indicate the segmentation mask."""

LLAMA4_VQA_PROMPT = """You are a helpful multimodal AI assistant powered by Llama-4-Scout-17B-16E-Instruct that can analyze images and answer questions about visual content. Provide accurate, detailed responses based on what you observe in the images."""

# 質問テンプレート
SHORT_QUESTION_LIST = [
    "Can you segment the {class_name} in this image?",
    "Please segment the {class_name}.",
    "Where is the {class_name}? Please segment it.",
    "Can you identify and segment the {class_name}?",
    "Please provide a segmentation mask for the {class_name}.",
]

LONG_QUESTION_LIST = [
    "Can you segment the region described as: {sent}?",
    "Please segment the area that matches: {sent}",
    "Where is the region that {sent}? Please segment it.",
    "Can you identify and segment the area described as: {sent}?",
    "Please provide a segmentation mask for the region: {sent}",
]

EXPLANATORY_QUESTION_LIST = [
    "Can you explain why this region is important?",
    "What makes this area significant?",
    "Why should we focus on this region?",
    "What is special about this area?",
    "Can you provide reasoning for this segmentation?",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

# 画像前処理の定数
# SAM用設定（変更なし）
SAM_PIXEL_MEAN = [123.675, 116.28, 103.53]
SAM_PIXEL_STD = [58.395, 57.12, 57.375]
SAM_IMAGE_SIZE = 1024

# Llama-4-Scout-17B-16E-Instruct用設定（公式仕様準拠）
LLAMA4_IMAGE_SIZE = 1120  # 公式最大解像度 1120x1120

# デフォルト設定
DEFAULT_IGNORE_LABEL = 255
DEFAULT_NUM_CLASSES_PER_SAMPLE = 3
DEFAULT_SAMPLES_PER_EPOCH = 500 * 8 * 2 * 10 