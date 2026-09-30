# NOTES

セッションを跨いで保持する実験の文脈。新しい記録は各節の先頭に追記する。

## 仮説

- exp002（human, 2026-09-30）: CatBoost に変え、モデルの seed 平均で予測を安定させると CV が改善する
- exp001（human, 2026-09-30）: カテゴリ列をそのまま渡せる LightGBM で、Cabin 分解と支出合計を加えたベースラインを作る

## 結論

- exp002（2026-09-30）: CV 0.8172 ± 0.0037。exp001 より +0.0013 だが fold の標準偏差より小さい。notebook 04 で見えていた OOF 0.8214 は fold 分割 seed も変えて平均した値で、固定分割では再現しない。採否は未判断
- exp001（2026-09-30）: CV 0.8159 ± 0.0050。notebook 03 の結果を完全に再現した。比較基準とする
- 移行前（notebook 時代）: FamilySize、NoSpend、SpendCount の追加と、CryoSleep による論理補完はいずれも CV か LB が下がり不採用（詳細は exp001/config.yaml のコメント）

## 次の一手

人間が決める。

## 提案

Claude Code の改善案。採否は人間が判断する。

- exp001 と exp002 の確率平均。OOF は 0.8178 で単体より高く、両者の OOF 予測の一致率は 94.6% なので誤り方が一部異なる。アンサンブル用の実験として起こす価値がある
- LB の自動記入。`make submit` は提出メッセージに実験 ID を入れているため、`kaggle competitions submissions` から `lb.public` を results.json に書き写すスクリプトを作れる。ルール上 lb は人間の記入なので、導入するかは判断を仰ぐ
- CV 分割のグループ化。PassengerId の先頭4桁は同行グループで、同じグループが train と valid に分かれると CV が楽観的になる可能性がある。変えると既存実験の CV と比較できなくなるため、導入するなら全実験を再評価する前提になる
- exp001/exp002 の lb は未記入。notebook 時代の提出（最高 0.80851）はどの版の notebook か対応が取れないため記入していない。LB が必要なら `make submit EXP=exp002` で提出して記入する
