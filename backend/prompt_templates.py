from langchain.prompts import PromptTemplate  # pyright: ignore[reportMissingImports]

# 1. 基本的なプロンプト
basic_prompt = PromptTemplate(
    template="""以下の文脈を使用して質問に答えてください。

文脈:
{context}

質問: {question}

回答:""",
    input_variables=["context", "question"]
)

# 2. 明示的な指示を含むプロンプト
explicit_prompt = PromptTemplate