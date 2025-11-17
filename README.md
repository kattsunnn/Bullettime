# Bullettime algorithm

```mermaid
flowchart TD
    input[多視点全方位画像] -->|全方位画像| gen_ppis[透視投影画像群の<br>生成]
    gen_ppis -->|透視投影画像群| pd1{骨格検出}
    pd1 -->|有| get_gaze_point[注視点を取得]
    pd1 -->|無| e1[終了]
    get_gaze_point -->|注視点座標| grouping[注視点をグルーピング]
    grouping -->|グルーピングした注視点の中心座標| gen_gaze_ppi[透視投影画像の生成]
    gen_gaze_ppi -->|透視投影画像| pd2{骨格検出}
    pd2 -->|有| scaling[スケーリング]
    pd2 -->|無| e1[終了]
    scaling -->|スケーリングした<br>透視投影画像| matching{視点間の<br>特徴点マッチング}
    matching -->|マッチング有| gen_bullettime[バレットタイム映像]
    matching -->|マッチング無| e1[終了]