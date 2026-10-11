# 盲標審查（blind_audit）

目的與設計見 `PREREGISTRATION.md`（標註前固定）。給標註者的說明是 `README_標註者說明.md`。

| 檔案 | 用途 | 給標註者？ |
|---|---|---|
| `form_for_annotators/` | 表單：`index.html` + `img/`（每題一張全圖、一張局部圖，只有紅框，沒有標籤、COCO 框、模型名稱） | 是（整個資料夾） |
| `README_標註者說明.md` | 判斷原則與操作 | 是 |
| `answer_key_PRIVATE.json` | 每題的來源（待檢、正對照、負對照）、模型標籤、框 | **否**；兩位標註者都交件之前，作者也不看 |
| `PREREGISTRATION.md` | 預先固定的題目組成、映射、分析、判讀規則 | 否（不必給標註者，避免透露假設） |
| `build_blind_audit.py` + `form_template.html` | 產生表單（需要 COCO 圖片，在 NDHU 上執行；種子 20261011） | 否 |
| `analyze_blind_audit.py` | 預先固定的分析；`--selftest` 用模擬標註者驗證 | 否 |

## 流程
1. 把 `form_for_annotators/` 整個資料夾與 `README_標註者說明.md` 交給兩位標註者（兩位都不是定標註規則的人，也不知道研究假設）。用瀏覽器開 `index.html`，填代號（A1、A2）。
2. 兩位各自獨立標完，按「匯出 CSV」，檔名為 `blind_labels_<代號>.csv`，寄回。
3. 兩份都收到後才執行分析：
   `python analyze_blind_audit.py answer_key_PRIVATE.json ../human_review/human_labels_final.csv blind_result.json blind_labels_A1.csv blind_labels_A2.csv`
4. 命名一致性：`blind_result_naming.csv` 交給 text-only judge。

## 注意
- `form_for_annotators/img/` 含 COCO 圖片（Flickr 逐圖授權），**不要公開或放進 GitHub**；公開只放程式、預先註冊文件、說明與分析結果。
- 負對照（平移）可能碰巧落在 COCO 沒標的真物體上，是近似的負對照，分析時要註明。
- 表單的進度存在瀏覽器 localStorage，請固定使用同一個瀏覽器，不要清除瀏覽資料。
