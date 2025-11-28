# Bullettime algorithm

```mermaid
flowchart TD
    input(多視点全方位画像) -->|全方位画像| gen_ppis[透視投影画像群の<br>生成]
    gen_ppis -->|透視投影画像群| pd1{骨格検出}
    pd1 -->|有| get_gaze_point[注視点を取得]
    pd1 -->|無| e1[終了]
    get_gaze_point -->|注視点座標群| grouping[注視点をグルーピング]
    grouping -->|グルーピングした注視点の中心座標| gen_gaze_ppi[注視画像の生成]
    gen_gaze_ppi -->|注視画像| scaling[スケーリング]
    scaling -->|スケーリングした注視画像| pd2{骨格検出}
    pd2 -->|有| matching{視点間の<br>特徴点マッチング}
    pd2 -->|無| e1[終了]
    matching -->|マッチング有| gen_bullettime(バレットタイム映像)
    matching -->|マッチング無| e1[終了]

    flowchart TD
    input[多視点全方位画像] -->|全方位画像| gen_ppis[透視投影画像群の<br>生成]
    gen_ppis -->|透視投影画像群| pd1{骨格検出}
    pd1 -->|有| get_gaze_point[注視点を取得]
    pd1 -->|無| e1[終了]
    get_gaze_point -->|注視点座標| grouping[注視点をグルーピング]
    grouping -->|グルーピングした注視点の中心座標| gen_gaze_ppi[透視投影画像の生成]
    gen_gaze_ppi -->|透視投影画像| gen_crop[局所画像の生成]
    gen_crop --> similar_img_search{類似画像検索}
    similar_img_search --> 3d_reconstruction[3次元復元]
    similar_img_search --> e1
    3d_reconstruction -->|透視投影画像の生成、スケーリング| scaling[バレットタイム画像の生成]