### HRNet彩色關節點系統
本專案在原始 HRNet 模型的基礎上，針對視覺化部分進行了重要改進：

- **彩色關節點系統**：原本的系統僅提供基本的關節點標示，本專案實現了完整的彩色化視覺化系統
  - 不同身體部位使用不同顏色系（頭部、上半身、下半身分別用不同色系）
  - 關節點大小根據置信度分數動態調整
  - 三層立體效果繪製（外圈、主體、內圈），增強視覺效果
- **增強版骨架連線**：自動連接相關關節點，不同身體部位使用不同顏色的連線
- **統計資訊面板**：顯示偵測統計資訊（有效關節點數、平均分數等），方便結果分析

---

## 環境需求

### 硬體需求
- **Kneron KL730** 開發板
- USB 連接線（建議使用 USB 3.0 Super Speed）

### 軟體需求
- Python 3.7+
- Kneron SDK (kp)
- OpenCV (cv2)
- NumPy

### 依賴套件安裝

```bash
pip install kneron-sdk opencv-python numpy
```

---

## 專案結構

```
final project HRNet/
├── README.md                          # 本文件
├── kneopi-examples-main/
│   └── ai_application/
│       └── plus_python/
│           ├── KL730HRNet.py          # 主程式
│           ├── res/
│           │   ├── firmware/
│           │   │   └── KL730/
│           │   │       └── kp_firmware.tar
│           │   ├── models/
│           │   │   └── KL730/
│           │   │       └── HRNet/
│           │   │           └── models_730.nef
│           │   └── images/            # 測試圖片
│           └── utils/                  # SDK 工具函數
└── models_730.nef                     # 模型檔案（備用位置）
```

---

## 快速開始

### 1. 連接設備

確保 KL730 開發板已透過 USB 連接到電腦，並確認設備已被識別。

### 2. 基本使用

```bash
# 進入 plus_python 目錄
cd kneopi-examples-main/ai_application/plus_python

# 執行腳本（使用預設參數）
python KL730HRNet.py -img test1.jpg

```

### 3. 查看結果

執行完成後，會生成 `hrnet_result.jpg` 檔案，包含：
- 原始圖片
- 17 個彩色關節點
- 骨架連線
- 統計資訊面板（可選）

---

## 使用說明

### 命令列參數

#### 基本參數
```bash
-p, --port_id          USB 端口 ID (預設: 0)
-fw, --firmware        韌體檔案路徑 (自動尋找)
-m, --model            模型檔案路徑 (自動尋找)
-img, --image          輸入圖片路徑 (預設: people.jpg)
```

#### 偵測參數
```bash
--detect_method        解碼方法 (argmax/subpixel/weighted/gaussian)
                       - argmax: 簡單快速 (預設)
                       - subpixel: 亞像素精度優化
                       - weighted: Top-K 加權平均
                       - gaussian: 高斯擬合優化
--thresh               分數閾值 (預設: 0.2)
```

#### 視覺化參數
```bash
--show_skeleton        顯示骨架連線 (預設: 開啟)
--no_skeleton          不顯示骨架連線
--show_labels          顯示關節點名稱標籤
--show_scores          顯示關節點分數
--point_size           關節點大小 (3-10, 預設: 5)
--line_thickness       骨架線條粗細 (1-5, 預設: 3)
--show_stats           顯示統計資訊面板 (預設: 開啟)
--no_stats             不顯示統計資訊面板
```
---

## 功能說明

### 關節點定義（17個）

系統偵測的 17 個 COCO 格式關節點：

```
0: nose (鼻子)
1: left_eye (左眼)
2: right_eye (右眼)
3: left_ear (左耳)
4: right_ear (右耳)
5: left_shoulder (左肩)
6: right_shoulder (右肩)
7: left_elbow (左手肘)
8: right_elbow (右手肘)
9: left_wrist (左手腕)
10: right_wrist (右手腕)
11: left_hip (左臀)
12: right_hip (右臀)
13: left_knee (左膝)
14: right_knee (右膝)
15: left_ankle (左腳踝)
16: right_ankle (右腳踝)
```

### 解碼方法比較

| 方法 | 速度 | 精度 | 適用場景 |
|------|------|------|----------|
| **argmax** | 5/5 | 3/5 | 即時應用、速度優先 |
| **subpixel** | 4/5 | 4/5 | 平衡速度與精度 |
| **weighted** | 3/5 | 4/5 | 多雜訊場景 |
| **gaussian** | 2/5 | 5/5 | 高精度需求 |

### 視覺化特色（本專案改進）

本專案在視覺化方面進行了重要改進，原本的系統僅提供基本的關節點標示，本專案實現了完整的彩色化視覺化系統：

- **彩色關節點系統**（本專案新增）
  - 不同身體部位使用不同顏色系，提升視覺辨識度
    - 頭部：藍綠色系
    - 上半身左：藍色系
    - 上半身右：綠色系
    - 下半身左：紅色系
    - 下半身右：紫色系
  - 關節點大小根據置信度分數動態調整（分數越高，節點越大）
  - 三層立體效果繪製（外圈白色、主體彩色、內圈深色），增強視覺層次

- **增強版骨架連線**（本專案改進）
  - 自動連接相關關節點，形成完整骨架
  - 不同身體部位使用不同顏色的連線（頭部、軀幹、手臂、腿部分別用不同顏色）

- **統計資訊面板**（本專案新增）
  - 顯示有效關節點數、平均分數、最高/最低分數等
  - 半透明背景設計，不遮擋原始圖片
  - 可選顯示關節點名稱標籤和置信度分數

---

## 輸出說明

### 終端輸出範例

```
[File Check]
  Firmware: /path/to/kp_firmware.tar
  Model: /path/to/models_730.nef
  Image: /path/to/test1.jpg

[Connect Device]
 - Success

[Upload Firmware]
 - Success

[Upload Model]
 - Success

[Read Image]
[Model Input] HxW = 256x192

[Inference]
 - Success

[Debug] heatmap shape: (17, 64, 48)

[Detection Method] 使用 argmax 方法

[Visualization] Skeleton=True, Labels=False, Scores=False

[Statistics]
  Valid Joints: 15/17
  Average Score (All): 0.4523
  Average Score (Valid): 0.5128
  Max Score: 0.9234
  Min Score (Valid): 0.2012
  Total Score: 7.6920

[Done] Saved: hrnet_result.jpg
```

### 輸出圖片

- **檔名**：`hrnet_result.jpg`

---

## 常見問題

### Q1: 找不到韌體或模型檔案？

**A:** 程式會自動尋找檔案，如果找不到，請使用 `-fw` 和 `-m` 參數手動指定路徑。

### Q2: USB 連接失敗？

**A:** 
- 確認 USB 線是否為 USB 3.0
- 確認設備驅動程式已正確安裝
- 嘗試不同的 USB 端口
- 檢查設備是否被其他程式占用

### Q3: 推論結果不準確？

**A:**
- 嘗試使用不同的解碼方法（`--detect_method`）
- 調整分數閾值（`--thresh`）
- 確認輸入圖片品質良好
- 確認圖片中人物清晰可見

### Q4: 執行速度慢？

**A:**
- 使用 `argmax` 方法（最快）
- 確認 USB 速度為 Super Speed
- 減少圖片解析度

---

## 程式結構

### 主要模組

1. **路徑處理模組**
   - `_find_kneopi_examples_path()` - 自動尋找專案目錄
   - `_get_default_firmware_path()` - 自動尋找韌體
   - `_get_default_model_path()` - 自動尋找模型

2. **工具函數模組**
   - `_get_model_input_hw()` - 讀取模型輸入尺寸
   - `_to_numpy()` - 轉換推論結果為 NumPy array

3. **關鍵點解碼模組**（4種方法）
   - `decode_heatmap_argmax()` - 簡單快速
   - `decode_heatmap_subpixel_refinement()` - 亞像素優化
   - `decode_heatmap_weighted_average()` - 加權平均
   - `decode_heatmap_gaussian_refinement()` - 高斯擬合

4. **視覺化模組**
   - `draw_keypoints_enhanced()` - 增強版視覺化
   - `get_skeleton_color()` - 骨架顏色配置

5. **主程式流程**
   - `main()` - 完整推理流程

---

## 技術細節

### 模型規格
- **輸入尺寸**：256×192（可自動偵測）
- **輸出格式**：Heatmap (17, H, W)
- **關節點數量**：17 個（COCO 格式）

### 依賴套件
- `kp` - Kneron SDK
- `cv2` (OpenCV) - 圖片處理
- `numpy` - 數值運算
- `argparse` - 命令列參數解析

---

## 授權

本專案基於 Kneron 範例程式碼修改，請參考原始授權條款。

---

## 參考資料

- [Kneron 官方文檔](https://www.kneron.com/)
- [HRNet 論文](https://arxiv.org/abs/1908.07919)
- [COCO 關鍵點格式](https://cocodataset.org/#format-data)


