from __future__ import annotations

from typing import Dict

from langchain.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]


# === Prompt Templates =======================================================

_BASIC_TEMPLATE = """あなたは質問応答アシスタントです。以下の【文脈】と【重要な指示】に厳密に従って、質問に答えてください。

【重要な指示】
1. 【文脈】に情報がある場合は、その情報だけを根拠として簡潔かつ具体的に答えてください。技名・術式・特殊能力・特徴・使用条件などの細部が示されている場合は、可能な限り列挙してください。
2. 複数の情報源が関連している場合は、それらを統合し、矛盾がない形で回答に反映してください。
3. 回答には、事実に基づかない前置きや丁寧すぎる言い回し（「お答えしますと〜」など）は含めないでください。

###
【文脈】
{context}
###

【質問】
{question}

【回答】"""

_PDF_TEMPLATE = """あなたは与えられたPDFの内容を詳しく説明できるアシスタントです。以下の指示に必ず従ってください。

1. 文脈に記載されている内容のみを使用する
2. 不明な点は推測せず「情報が見つかりませんでした」と答える
3. 箇条書きで整理できる場合は積極的に活用する

===
文脈:
{context}
===

質問: {question}

回答:"""

_EXPLAINER_TEMPLATE = """以下の文脈を参考に、RAGシステムの構成や振る舞いをわかりやすく説明してください。

Context:
{context}

Question: {question}

Answer:"""


# === Prompt Instances =======================================================

BASIC_PROMPT = PromptTemplate(
    template=_BASIC_TEMPLATE,
    input_variables=["context", "question"],
)

PDF_PROMPT = PromptTemplate(
    template=_PDF_TEMPLATE,
    input_variables=["context", "question"],
)

EXPLAINER_PROMPT = PromptTemplate(
    template=_EXPLAINER_TEMPLATE,
    input_variables=["context", "question"],
)

PROMPTS: Dict[str, PromptTemplate] = {
    "basic": BASIC_PROMPT,
    "pdf": PDF_PROMPT,
    "explainer": EXPLAINER_PROMPT,
}


def get_prompt(name: str = "basic") -> PromptTemplate:
    """
    指定された名前に対応する PromptTemplate を返す。
    未定義の名前が渡された場合は BASIC_PROMPT を返す。
    """

    return PROMPTS.get(name, BASIC_PROMPT)