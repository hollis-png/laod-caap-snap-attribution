# E5 第二標註者
- 開啟 `index.html`（雙擊即可，不需伺服器；圖在 `img/`）。規則與第一位標註者相同，頁面不含任何 AI／第一位標註者判定。
- 至少標 R001–R100，完成後按「匯出 CSV」，得到 `human_labels_annotator2.csv`。
- 算 human–human κ：
  `python3 ../human_review/compute_agreement_hh.py ../human_review/human_labels.csv human_labels_annotator2.csv`
  （第一位用「第一輪」未經 A2 修改的標記，維持獨立。）
- 答案檔 `answer_key_DO_NOT_OPEN_BEFORE_REVIEW.json` 沒有放進這個資料夾，傳給別人時只傳本資料夾。
