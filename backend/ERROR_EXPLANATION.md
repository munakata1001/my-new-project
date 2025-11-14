# Gemini API エラーの詳細解説

## エラーメッセージ

```
404 models/gemini-pro is not found for API version v1beta, 
or is not supported for generateContent. 
Call ListModels to see the list of available models and their supported methods.
```

## エラーの根本原因

### 1. **モデル名の非存在**
- `gemini-pro`、`gemini-1.5-pro`、`gemini-1.5-flash` は**現在のGoogle Gemini APIでは利用できません**
- Googleは2024年後半から2025年にかけて、これらのモデルを新しいバージョンに置き換えました

### 2. **APIバージョンの問題**
- `langchain_google_genai`ライブラリは内部的に**v1beta API**を使用しています
- v1beta APIでは、古いモデル名（`gemini-pro`など）はサポートされていません
- 新しいモデル（Gemini 2.0/2.5シリーズ）のみが利用可能です

### 3. **モデル名の形式**
- 実際のモデル名は `models/gemini-2.5-flash` のような形式です
- しかし、`langchain_google_genai`は自動的に `models/` プレフィックスを追加するため、`gemini-2.5-flash` と指定すればOKです

## 利用可能なモデル一覧（2025年1月時点）

### 安定版（推奨）
- ✅ **`gemini-2.5-flash`** - 高速で安定したモデル（推奨）
- ✅ **`gemini-2.5-pro`** - 高精度モデル
- ✅ **`gemini-2.0-flash`** - 安定版の高速モデル

### 最新版
- ✅ **`gemini-flash-latest`** - 最新のFlashモデル
- ✅ **`gemini-pro-latest`** - 最新のProモデル

### プレビュー版
- ⚠️ **`gemini-2.5-flash-preview-*`** - プレビュー版（実験的）

## 修正内容

### 変更前（エラーが発生）
```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-pro",  # ❌ このモデルは存在しない
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

### 変更後（修正済み）
```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ 利用可能なモデル
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

## エラーが続く場合の対処法

### 1. **サーバーの再起動**
- FastAPIサーバーを完全に停止して再起動してください
- 古いコードがメモリに残っている可能性があります

### 2. **Pythonキャッシュのクリア**
```powershell
# PowerShellで実行
Remove-Item -Recurse -Force __pycache__
```

### 3. **コードの確認**
- `basic_rag_system.py`の28行目と99行目が `gemini-2.5-flash` になっているか確認
- 他のファイルで `gemini-pro` が使われていないか確認

### 4. **環境変数の確認**
- `GOOGLE_API_KEY`が正しく設定されているか確認
- `.env`ファイルにAPIキーが記載されているか確認

## 技術的な詳細

### APIバージョンについて
- **v1beta**: ベータ版API（一部のモデルのみサポート）
- **v1**: 安定版API（すべてのモデルをサポート）

### モデル名の変換
`langchain_google_genai`は内部的に以下の変換を行います：
- 入力: `gemini-2.5-flash`
- 実際のAPI呼び出し: `models/gemini-2.5-flash`

### エラーの流れ
1. コードで `gemini-pro` を指定
2. `langchain_google_genai`が `models/gemini-pro` に変換
3. v1beta APIにリクエスト送信
4. APIが「404 モデルが見つかりません」と返す
5. エラーメッセージが表示される

## まとめ

このエラーは、**古いモデル名を使用している**ことが原因です。新しいモデル名（`gemini-2.5-flash`など）に変更することで解決します。

**重要なポイント：**
- ✅ `gemini-2.5-flash` を使用（推奨）
- ❌ `gemini-pro` は使用不可
- ❌ `gemini-1.5-pro` は使用不可
- ❌ `gemini-1.5-flash` は使用不可

