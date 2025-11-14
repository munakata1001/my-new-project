from __future__ import annotations

from typing import Dict

from langchain.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]


# === Prompt Templates =======================================================

_BASIC_TEMPLATE = """あなたは信頼できるアシスタントです。以下の文脈のみを根拠に質問へ日本語で答えてください。

###
文脈:
{context}
###

質問: {question}

回答:"""

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