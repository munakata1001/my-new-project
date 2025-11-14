# Gemini API エラーの詳細解説（完全版）

## エラーメッセージの分析

あなたが受け取ったエラーメッセージ：

```
1. 404 models/gemini-1.5-pro is not found for API version v1beta
2. 404 models/gemini-1.5-flash is not found for API version v1beta  
3. 404 models/gemini-pro is not found for API version v1beta
```

### なぜ3つの異なるエラーが出るのか？

これは、**サーバーが再起動されていない**可能性が高いです。以下のいずれかが原因です：

1. **古いコードがメモリに残っている**
   - FastAPIサーバーが古いコードを実行している
   - Pythonの`__pycache__`に古いコンパイル済みコードが残っている

2. **複数のリクエストで異なるモデル名が試されている**
   - サーバー起動時に異なるモデル名で初期化が試みられた
   - エラーハンドリングで複数のモデル名が試された

## エラーの根本原因

### 1. モデル名の非存在（最重要）

以下のモデル名は**現在のGoogle Gemini APIでは利用できません**：

| モデル名 | ステータス | 理由 |
|---------|---------|------|
| `gemini-pro` | ❌ 利用不可 | 2024年に廃止 |
| `gemini-1.5-pro` | ❌ 利用不可 | 2024年に廃止 |
| `gemini-1.5-flash` | ❌ 利用不可 | 2024年に廃止 |

**Googleは2024年後半から2025年にかけて、これらのモデルを新しいバージョンに置き換えました。**

### 2. APIバージョンの問題

```
API version v1beta
```

- `langchain_google_genai`ライブラリは内部的に**v1beta API**を使用
- v1beta APIでは、古いモデル名はサポートされていません
- 新しいモデル（Gemini 2.0/2.5シリーズ）のみが利用可能

### 3. モデル名の形式

実際のAPIでは、モデル名は以下の形式です：
- 完全な形式: `models/gemini-2.5-flash`
- `langchain_google_genai`は自動的に`models/`プレフィックスを追加
- そのため、コードでは`gemini-2.5-flash`と指定すればOK

## 利用可能なモデル一覧（2025年1月時点）

### ✅ 安定版（推奨）

| モデル名 | 説明 | 推奨度 |
|---------|------|--------|
| `gemini-2.5-flash` | 高速で安定したモデル | ⭐⭐⭐⭐⭐ |
| `gemini-2.5-pro` | 高精度モデル | ⭐⭐⭐⭐ |
| `gemini-2.0-flash` | 安定版の高速モデル | ⭐⭐⭐⭐ |

### ✅ 最新版

| モデル名 | 説明 | 推奨度 |
|---------|------|--------|
| `gemini-flash-latest` | 最新のFlashモデル | ⭐⭐⭐ |
| `gemini-pro-latest` | 最新のProモデル | ⭐⭐⭐ |

### ⚠️ プレビュー版（実験的）

| モデル名 | 説明 | 推奨度 |
|---------|------|--------|
| `gemini-2.5-flash-preview-*` | プレビュー版 | ⭐⭐ |

## 現在のコードの状態

### ✅ 修正済み（basic_rag_system.py）

```python
# SimpleRAGクラス（28行目）
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ 正しいモデル名
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# PDFRagSystemクラス（99行目）
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # ✅ 正しいモデル名
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
```

## エラーが続く場合の対処法

### 🔴 ステップ1: サーバーの完全停止

1. **FastAPIサーバーを完全に停止**
   - ターミナルで`Ctrl+C`を押す
   - プロセスが完全に終了するまで待つ

2. **実行中のプロセスを確認**
   ```powershell
   # PowerShellで実行
   Get-Process python | Where-Object {$_.Path -like "*RAG*"}
   ```

3. **プロセスを強制終了（必要に応じて）**
   ```powershell
   Stop-Process -Name python -Force
   ```

### 🟡 ステップ2: キャッシュのクリア（完了済み）

```powershell
# Pythonキャッシュを削除
Remove-Item -Recurse -Force __pycache__
```

### 🟢 ステップ3: サーバーの再起動

```powershell
# サーバーを再起動
cd my-new-project/backend
uvicorn main:app --reload
```

### 🔵 ステップ4: 動作確認

1. サーバーが正常に起動したか確認
2. ブラウザで`http://localhost:8000`にアクセス
3. フロントエンドから質問を送信してテスト

## 技術的な詳細

### APIバージョンについて

```
v1beta API
  ├─ 古いモデル（gemini-pro等）: ❌ サポートなし
  └─ 新しいモデル（gemini-2.5-*）: ✅ サポートあり

v1 API（安定版）
  ├─ すべてのモデル: ✅ サポートあり
  └─ ただし、langchain_google_genaiはv1betaを使用
```

### モデル名の変換プロセス

```
コードで指定: gemini-2.5-flash
  ↓
langchain_google_genai内部処理
  ↓
API呼び出し: models/gemini-2.5-flash
  ↓
Google Gemini API
  ↓
レスポンス: 成功 ✅
```

### エラーの発生フロー

```
1. コードで古いモデル名を指定（例: gemini-pro）
   ↓
2. langchain_google_genaiが models/gemini-pro に変換
   ↓
3. v1beta APIにリクエスト送信
   ↓
4. APIが「404 モデルが見つかりません」と返す
   ↓
5. エラーメッセージが表示される
```

## よくある質問（FAQ）

### Q1: なぜコードを修正したのにエラーが出るの？

**A:** サーバーが再起動されていないため、古いコードがメモリに残っています。

**解決策:** サーバーを完全に停止して再起動してください。

### Q2: どのモデルを使えばいいの？

**A:** `gemini-2.5-flash`を推奨します。高速で安定しており、コスト効率も良いです。

### Q3: 他のモデルも試したい

**A:** 以下のモデル名を試せます：
- `gemini-2.5-pro`（高精度が必要な場合）
- `gemini-flash-latest`（最新機能が必要な場合）

### Q4: エラーが続く場合は？

**A:** 以下を確認してください：
1. `GOOGLE_API_KEY`が正しく設定されているか
2. `.env`ファイルにAPIキーが記載されているか
3. インターネット接続が正常か
4. Google Gemini APIの利用制限に達していないか

## まとめ

### ✅ 修正済み
- コードは`gemini-2.5-flash`に更新済み
- Pythonキャッシュはクリア済み

### 🔴 必要な作業
- **サーバーの完全停止と再起動**（最重要）

### 📝 チェックリスト

- [ ] FastAPIサーバーを完全に停止
- [ ] `__pycache__`フォルダを削除（完了済み）
- [ ] サーバーを再起動
- [ ] 動作確認

## 参考情報

- [Google Gemini API ドキュメント](https://ai.google.dev/docs)
- [LangChain Google Generative AI ドキュメント](https://python.langchain.com/docs/integrations/llms/google_generative_ai)

---

**重要:** サーバーを再起動しない限り、エラーは解消されません。必ずサーバーを再起動してください。

