# judge 重新分類：狀態（2026-10-09，已完成）

三組各 775/775，無解析失敗；結果見 `REPORT.md`、`REPORT_binary.md`。以下為先前中斷時的記錄，僅供追溯。


**停在 OpenRouter 額度用盡**（HTTP 402 "Insufficient credits. This account never purchased credits"）。

| run | 已分類 | 備註 |
|---|---|---|
| `out_deepseek_v1.jsonl`（deepseek-v4-flash-0731，prompt v1） | 168 / 775 | 其中 9 筆空回覆（max_tokens=300 被隱藏推理用光，需用 ≥3000 重試） |
| `out_deepseek_v2.jsonl`（同模型，prompt v2） | 170 / 775 | 12 筆空回覆，同上 |
| `out_mistral_v1.jsonl`（mistral-large-4-0） | 42 / 775 | 其中 23 筆無效，已不可用 |

`raw_full.json` 依偵測數由大到小排列，所以已完成的是**最常見的 168 個標籤**，不是隨機子集。

## 初步（不能當結論）
- deepseek v1 對你的 B1 人工標籤：50 個中已涵蓋 36 個，一致 34；排除 prompt 範例詞後 24/24 一致。不一致 2 個：surfer（judge 較細粒度、你同義詞）、flowers（judge 非本體、你較細粒度）。
- v1 與 v2（不同定義寫法、不同範例）在 154 個共同標籤上一致 152 個。
- **偏誤：** 涵蓋的是高頻（多半較好判）的標籤，且空回覆（可能是較難的）被排除，所以一致率偏高。事先寫定的決策規則（非範例標籤 κ ≥ 0.6 且 v1–v2 κ ≥ 0.6）需要完整結果才能判定。

## 續跑前要改
1. `run_judge.py` 的 `max_tokens` 由 300 改成 ≥ 3000（DeepSeek v4 的隱藏推理最多用到約 1,400 token）。
2. 重跑 `parsed is None` 的標籤（腳本可續跑，但目前會把已存在的空回覆視為完成，需先刪除那些列）。
3. 取得可用額度或換成其他 judge。
