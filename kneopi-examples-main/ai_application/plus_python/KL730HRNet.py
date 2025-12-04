# ****************************************************************************
#  HRNet Pose demo on KNEO Pi (KL730)
#  人體姿態估計 - 用 HRNet 模型偵測人體關節點
#  
#  流程：讀圖 → 推論 → 取出 heatmap → 解碼關節點 → 畫骨架
#
#  使用範例：
#  python KL730HRNet.py -img test1.jpg -p 0
# ****************************************************************************

import os
import sys
import argparse

# 取得腳本所在目錄（用來找檔案路徑）
PWD = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 路徑處理相關函數（因為腳本可能在不同位置執行）
# ============================================================================

def _find_kneopi_examples_path():
    """
    自動找 kneopi-examples-main 目錄在哪
    因為腳本可能在專案根目錄或 plus_python 目錄執行
    """
    current = PWD
    # 檢查當前目錄是否在 plus_python 中
    if os.path.basename(current) == 'plus_python':
        # 檢查父目錄結構
        parent = os.path.dirname(current)
        if os.path.basename(parent) == 'ai_application':
            grandparent = os.path.dirname(parent)
            if os.path.basename(grandparent) == 'kneopi-examples-main':
                return grandparent
    
    # 檢查當前目錄是否有 kneopi-examples-main 子目錄
    kneopi_path = os.path.join(current, 'kneopi-examples-main')
    if os.path.exists(kneopi_path):
        return kneopi_path
    
    # 向上查找 kneopi-examples-main
    check_dir = current
    for _ in range(5):  # 最多向上查找5層
        kneopi_path = os.path.join(check_dir, 'kneopi-examples-main')
        if os.path.exists(kneopi_path):
            return kneopi_path
        parent = os.path.dirname(check_dir)
        if parent == check_dir:  # 到達根目錄
            break
        check_dir = parent
    
    return None

# 添加 kneron SDK 路徑
kneopi_root = _find_kneopi_examples_path()
if kneopi_root:
    kneron_sdk_path = os.path.join(kneopi_root, 'ai_application', 'plus_python')
    if os.path.exists(kneron_sdk_path):
        sys.path.insert(0, kneron_sdk_path)
        sys.path.insert(0, os.path.join(kneron_sdk_path, '..'))
else:
    # 如果找不到，嘗試直接使用當前目錄（假設腳本在 plus_python 中）
    kneron_sdk_path = PWD
    if os.path.exists(os.path.join(kneron_sdk_path, 'utils')):
        sys.path.insert(0, kneron_sdk_path)

try:
    from utils.ExampleHelper import get_device_usb_speed_by_port_id
except ImportError:
    # 如果找不到，嘗試直接定義一個簡單版本
    def get_device_usb_speed_by_port_id(usb_port_id):
        try:
            import kp
            return kp.UsbSpeed.KP_USB_SPEED_SUPER
        except:
            return None

import kp
import cv2
import numpy as np


def _get_model_input_hw(model_nef_descriptor):
    """
    從模型描述檔讀取輸入圖片的寬高
    如果讀不到就用預設值 256x192
    """
    try:
        # [N, C, H, W]
        shape = model_nef_descriptor.models[0].input_nodes[0].tensor_shape_info.v2.shape
        in_h, in_w = int(shape[2]), int(shape[3])
        return in_h, in_w
    except Exception:
        pass

    try:
        shape = model_nef_descriptor.models[0].input_nodes[0].tensor_shape_info.shape
        in_h, in_w = int(shape[2]), int(shape[3])
        return in_h, in_w
    except Exception:
        pass

    print('[Model Input] auto-detect failed, fallback to 256x192')
    return 256, 192


def _extract_shape(obj):
    """盡量從不同 SDK 欄位拿 shape."""
    for key in ("shape_list", "shape", "output_shape", "dims"):
        if hasattr(obj, key):
            try:
                shp = getattr(obj, key)
                return tuple(int(x) for x in list(shp))
            except Exception:
                pass
    return None


def _to_numpy(inf_node, debug=False):
    """
    把推論結果轉成 numpy array
    因為 SDK 版本不同，結果格式可能不一樣，所以試很多種方法
    """
    import numpy as _np

    # 已經是 ndarray 的情況
    if isinstance(inf_node, _np.ndarray):
        return inf_node.astype(_np.float32, copy=False)

    #  你這個 SDK 版本最重要的一行：結果在 .ndarray 裡
    if hasattr(inf_node, "ndarray"):
        try:
            return _np.asarray(inf_node.ndarray, dtype=_np.float32)
        except Exception:
            # 真的有問題才往下嘗試其他方式
            pass

    if debug:
        try:
            print('[DEBUG] node type:', type(inf_node))
            attrs = [a for a in dir(inf_node) if not a.startswith('__')]
            print('[DEBUG] attrs:', attrs)
        except Exception:
            pass

    # 1) methods that directly give numpy-like
    for meth in ("to_numpy", "numpy", "as_numpy"):
        if hasattr(inf_node, meth) and callable(getattr(inf_node, meth)):
            try:
                arr = getattr(inf_node, meth)()
                return _np.asarray(arr, dtype=_np.float32)
            except Exception:
                pass

    # 2) float_data + shape
    if hasattr(inf_node, 'float_data'):
        data = getattr(inf_node, 'float_data')
        arr = _np.asarray(list(data), dtype=_np.float32)
        shp = _extract_shape(inf_node)
        if shp:
            try:
                arr = arr.reshape(shp)
            except Exception:
                pass
        return arr

    # 3) data list/tuple
    if hasattr(inf_node, 'data') and isinstance(getattr(inf_node, 'data'), (list, tuple)):
        data = getattr(inf_node, 'data')
        arr = _np.asarray(list(data), dtype=_np.float32)
        shp = _extract_shape(inf_node)
        if shp:
            try:
                arr = arr.reshape(shp)
            except Exception:
                pass
        return arr

    # 4) data bytes + shape
    if hasattr(inf_node, 'data') and not isinstance(getattr(inf_node, 'data'), (list, tuple)):
        try:
            buf = _np.frombuffer(getattr(inf_node, 'data'), dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    buf = buf.reshape(shp)
                except Exception:
                    pass
            return buf
        except Exception:
            pass

    # 5) buffer bytes + shape
    if hasattr(inf_node, 'buffer'):
        try:
            buf = _np.frombuffer(getattr(inf_node, 'buffer'), dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    buf = buf.reshape(shp)
                except Exception:
                    pass
            return buf
        except Exception:
            pass

    # 6) iterable fallback
    if hasattr(inf_node, '__len__') and hasattr(inf_node, '__getitem__'):
        try:
            arr = _np.asarray([inf_node[i] for i in range(len(inf_node))], dtype=_np.float32)
            shp = _extract_shape(inf_node)
            if shp:
                try:
                    arr = arr.reshape(shp)
                except Exception:
                    pass
            return arr
        except Exception:
            pass

    # 7) 各種 getter
    for getter in ("get_float_data", "get_data", "get_buffer"):
        if hasattr(inf_node, getter) and callable(getattr(inf_node, getter)):
            try:
                raw = getattr(inf_node, getter)()
                if isinstance(raw, (bytes, bytearray)):
                    arr = _np.frombuffer(raw, dtype=_np.float32)
                else:
                    arr = _np.asarray(list(raw), dtype=_np.float32)
                shp = _extract_shape(inf_node)
                if shp:
                    try:
                        arr = arr.reshape(shp)
                    except Exception:
                        pass
                return arr
            except Exception:
                pass

    raise TypeError('Unknown inference node output type; cannot convert to numpy.')

def decode_heatmap_argmax(heatmap_single):
    """
    方法1: 簡單 argmax 偵測
    直接找 heatmap 裡最大值的位置，最快但精度較低
    """
    Hh, Wh = heatmap_single.shape
    flat = heatmap_single.ravel()
    idx = flat.argmax()
    y = float(idx // Wh)
    x = float(idx % Wh)
    score = float(flat[idx])
    return x, y, score


def decode_heatmap_subpixel_refinement(heatmap_single):
    """
    方法2: 亞像素精度優化
    用周圍像素的梯度來微調位置，比 argmax 準一點
    """
    Hh, Wh = heatmap_single.shape
    flat = heatmap_single.ravel()
    idx = flat.argmax()
    py = int(idx // Wh)
    px = int(idx % Wh)
    max_score = float(flat[idx])
    
    # 邊界檢查
    if px < 1 or px >= Wh - 1 or py < 1 or py >= Hh - 1:
        return float(px), float(py), max_score
    
    # 使用周圍 3x3 區域計算梯度，優化位置
    diff_x = heatmap_single[py, px+1] - heatmap_single[py, px-1]
    diff_y = heatmap_single[py+1, px] - heatmap_single[py-1, px]
    
    # 調整位置 (類似原始 HRNet 的後處理)
    x = float(px) + np.sign(diff_x) * 0.25
    y = float(py) + np.sign(diff_y) * 0.25
    
    return x, y, max_score


def decode_heatmap_weighted_average(heatmap_single, topk=9):
    """
    方法3: Top-K 加權平均
    取前 k 個高分的點，用分數當權重算平均位置，比較穩定
    """
    Hh, Wh = heatmap_single.shape
    flat = heatmap_single.ravel()
    
    # 獲取 top-k 位置
    topk_idx = np.argsort(flat)[-topk:]
    topk_scores = flat[topk_idx]
    
    # 計算權重 (使用 softmax-like 權重)
    topk_scores = np.maximum(topk_scores, 0)  # 確保非負
    weights = topk_scores / (topk_scores.sum() + 1e-8)
    
    # 轉換為座標
    ys = (topk_idx // Wh).astype(np.float32)
    xs = (topk_idx % Wh).astype(np.float32)
    
    # 加權平均
    x = float(np.sum(xs * weights))
    y = float(np.sum(ys * weights))
    score = float(np.max(topk_scores))
    
    return x, y, score


def decode_heatmap_gaussian_refinement(heatmap_single):
    """
    方法4: 高斯擬合優化
    用 3x3 區域算質心，最準但比較慢
    """
    Hh, Wh = heatmap_single.shape
    flat = heatmap_single.ravel()
    idx = flat.argmax()
    py = int(idx // Wh)
    px = int(idx % Wh)
    max_score = float(flat[idx])
    
    # 邊界檢查
    if px < 1 or px >= Wh - 1 or py < 1 or py >= Hh - 1:
        return float(px), float(py), max_score
    
    # 提取 3x3 區域
    patch = heatmap_single[py-1:py+2, px-1:px+2].copy()
    patch = np.maximum(patch, 0)  # 確保非負
    
    if patch.sum() < 1e-6:
        return float(px), float(py), max_score
    
    # 計算質心 (類似高斯中心)
    y_coords, x_coords = np.mgrid[0:3, 0:3]
    weights = patch / (patch.sum() + 1e-8)
    
    center_x = np.sum(x_coords * weights)
    center_y = np.sum(y_coords * weights)
    
    # 轉換回原座標系
    x = float(px - 1 + center_x)
    y = float(py - 1 + center_y)
    
    return x, y, max_score


# ============================================================================
# COCO 格式的關節點定義（17個點）
# ============================================================================
COCO_KEYPOINTS = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# 骨架連線定義：哪些關節點要連在一起 [起點索引, 終點索引]
COCO_SKELETON = [
    [15, 13],  # left_ankle -> left_knee
    [13, 11],  # left_knee -> left_hip
    [16, 14],  # right_ankle -> right_knee
    [14, 12],  # right_knee -> right_hip
    [11, 12],  # left_hip -> right_hip
    [5, 11],   # left_shoulder -> left_hip
    [6, 12],   # right_shoulder -> right_hip
    [5, 6],    # left_shoulder -> right_shoulder
    [5, 7],    # left_shoulder -> left_elbow
    [6, 8],    # right_shoulder -> right_elbow
    [7, 9],    # left_elbow -> left_wrist
    [8, 10],   # right_elbow -> right_wrist
    [5, 0],    # left_shoulder -> nose (簡化，實際通過頭部)
    [6, 0],    # right_shoulder -> nose
    [0, 1],    # nose -> left_eye
    [0, 2],    # nose -> right_eye
    [1, 3],    # left_eye -> left_ear
    [2, 4],    # right_eye -> right_ear
]

# 關節點的顏色（BGR格式，OpenCV用）
# 不同身體部位用不同顏色系，比較好看
JOINT_COLORS = {
    # 頭部 (藍綠色系)
    0: (255, 255, 0),    # nose - 青色
    1: (255, 200, 0),    # left_eye - 淺青色
    2: (255, 200, 0),    # right_eye - 淺青色
    3: (200, 255, 0),    # left_ear - 黃綠色
    4: (200, 255, 0),    # right_ear - 黃綠色
    # 上半身左 (藍色系)
    5: (255, 0, 0),      # left_shoulder - 藍色
    7: (200, 0, 0),      # left_elbow - 深藍色
    9: (150, 0, 0),      # left_wrist - 更深藍色
    # 上半身右 (綠色系)
    6: (0, 255, 0),      # right_shoulder - 綠色
    8: (0, 200, 0),      # right_elbow - 深綠色
    10: (0, 150, 0),     # right_wrist - 更深綠色
    # 下半身左 (紅色系)
    11: (0, 0, 255),     # left_hip - 紅色
    13: (0, 0, 200),     # left_knee - 深紅色
    15: (0, 0, 150),     # left_ankle - 更深紅色
    # 下半身右 (紫色系)
    12: (255, 0, 255),   # right_hip - 洋紅色
    14: (200, 0, 200),   # right_knee - 深洋紅色
    16: (150, 0, 150),   # right_ankle - 更深洋紅色
}

# 骨架連線的顏色（BGR格式）
SKELETON_COLORS = {
    'head': (200, 200, 200),      # 頭部連線 - 灰色
    'torso': (128, 128, 255),     # 軀幹 - 淺藍色
    'left_arm': (255, 128, 128),  # 左手 - 淺紅色
    'right_arm': (128, 255, 128), # 右手 - 淺綠色
    'left_leg': (255, 255, 128),  # 左腿 - 淺黃色
    'right_leg': (255, 128, 255), # 右腿 - 淺洋紅色
}


def get_skeleton_color(idx1, idx2):
    """
    根據兩個關節點的索引決定連線顏色
    不同身體部位用不同顏色
    """
    # 頭部 (0-4)
    if idx1 <= 4 or idx2 <= 4:
        return SKELETON_COLORS['head']
    # 軀幹 (5,6,11,12)
    if idx1 in [5, 6, 11, 12] and idx2 in [5, 6, 11, 12]:
        return SKELETON_COLORS['torso']
    # 左手 (5,7,9)
    if idx1 in [5, 7, 9] or idx2 in [5, 7, 9]:
        return SKELETON_COLORS['left_arm']
    # 右手 (6,8,10)
    if idx1 in [6, 8, 10] or idx2 in [6, 8, 10]:
        return SKELETON_COLORS['right_arm']
    # 左腿 (11,13,15)
    if idx1 in [11, 13, 15] or idx2 in [11, 13, 15]:
        return SKELETON_COLORS['left_leg']
    # 右腿 (12,14,16)
    if idx1 in [12, 14, 16] or idx2 in [12, 14, 16]:
        return SKELETON_COLORS['right_leg']
    return (200, 200, 200)  # 預設灰色


def draw_keypoints_enhanced(img, keypoints, threshold=0.2, 
                           show_skeleton=True, show_labels=False, 
                           show_scores=False, point_radius=5, 
                           line_thickness=3):
    """
    畫關節點和骨架的函數
    
    參數：
        img: 原始圖片（BGR格式）
        keypoints: 17個關節點的座標和分數 [(x, y, score), ...]
        threshold: 分數低於這個值就不畫
        show_skeleton: 要不要畫骨架連線
        show_labels: 要不要顯示關節點名稱（像 nose, left_eye 之類的）
        show_scores: 要不要顯示分數
        point_radius: 關節點的大小
        line_thickness: 骨架線條的粗細
    
    回傳：畫好的圖片
    """
    vis = img.copy()
    
    # 過濾低分數的關節點
    valid_kpts = []
    for i, (x, y, score) in enumerate(keypoints):
        if score >= threshold:
            valid_kpts.append((i, x, y, score))
    
    if not valid_kpts:
        return vis
    
    # 繪製骨架連線
    if show_skeleton:
        for (idx1, idx2) in COCO_SKELETON:
            if idx1 < len(keypoints) and idx2 < len(keypoints):
                x1, y1, s1 = keypoints[idx1]
                x2, y2, s2 = keypoints[idx2]
                
                # 只有兩個關節點都超過閾值才繪製連線
                if s1 >= threshold and s2 >= threshold:
                    pt1 = (int(x1), int(y1))
                    pt2 = (int(x2), int(y2))
                    color = get_skeleton_color(idx1, idx2)
                    cv2.line(vis, pt1, pt2, color, line_thickness, lineType=cv2.LINE_AA)
    
    # 繪製關節點
    for i, (x, y, score) in enumerate(keypoints):
        if score < threshold:
            continue
        
        pt = (int(x), int(y))
        color = JOINT_COLORS.get(i, (0, 255, 255))  # 預設黃色
        
        # 根據分數調整節點大小 (分數越高，節點越大)
        radius = int(point_radius * (0.7 + 0.6 * score))
        radius = max(3, min(radius, 12))  # 限制在 3-12 之間
        
        # 繪製外圈 (較亮)
        cv2.circle(vis, pt, radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        # 繪製主體
        cv2.circle(vis, pt, radius, color, -1, lineType=cv2.LINE_AA)
        # 繪製內圈 (較暗，增加立體感)
        cv2.circle(vis, pt, max(2, radius // 2), tuple(c // 2 for c in color), -1, lineType=cv2.LINE_AA)
        
        # 顯示標籤
        if show_labels and i < len(COCO_KEYPOINTS):
            label = COCO_KEYPOINTS[i]
            label_offset_x = radius + 8
            label_offset_y = -radius - 8
            label_pos = (int(x) + label_offset_x, int(y) + label_offset_y)
            
            # 計算文字大小（適配圖片大小）
            font_scale = 0.5  # 適中的字體大小
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # 半透明背景框（更柔和）
            padding = 4
            cv2.rectangle(vis, 
                         (label_pos[0] - padding, label_pos[1] - text_h - padding),
                         (label_pos[0] + text_w + padding, label_pos[1] + baseline + padding),
                         (0, 0, 0), -1)
            # 添加邊框線
            cv2.rectangle(vis, 
                         (label_pos[0] - padding, label_pos[1] - text_h - padding),
                         (label_pos[0] + text_w + padding, label_pos[1] + baseline + padding),
                         (255, 255, 255), 1)
            
            # 文字（白色，清晰易讀）
            cv2.putText(vis, label, label_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # 顯示分數
        if show_scores:
            score_text = f'{score:.2f}'
            score_offset_x = radius + 8
            score_offset_y = radius + 12
            score_pos = (int(x) + score_offset_x, int(y) + score_offset_y)
            
            # 計算文字大小（略小於標籤）
            font_scale = 0.45
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # 半透明背景框
            padding = 3
            cv2.rectangle(vis,
                         (score_pos[0] - padding, score_pos[1] - text_h - padding),
                         (score_pos[0] + text_w + padding, score_pos[1] + baseline + padding),
                         (0, 0, 0), -1)
            # 添加邊框線
            cv2.rectangle(vis,
                         (score_pos[0] - padding, score_pos[1] - text_h - padding),
                         (score_pos[0] + text_w + padding, score_pos[1] + baseline + padding),
                         (0, 255, 255), 1)
            
            # 文字（青色，與節點顏色協調）
            cv2.putText(vis, score_text, score_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    
    return vis


def decode_heatmap_to_keypoints(heatmap, orig_w, orig_h, method='argmax'):
    """
    從 heatmap 解碼出關節點座標
    
    參數：
        heatmap: 模型輸出的熱力圖 (K個關節點, H高, W寬)
        orig_w, orig_h: 原始圖片的寬高（用來縮放座標）
        method: 用哪種方法解碼 ('argmax', 'subpixel', 'weighted', 'gaussian')
    
    回傳：17個關節點的座標和分數 [(x, y, score), ...]
    """
    if heatmap.ndim == 4:
        heatmap = heatmap[0]
    
    K, Hh, Wh = heatmap.shape
    
    # 選擇偵測方法
    decode_funcs = {
        'argmax': decode_heatmap_argmax,
        'subpixel': decode_heatmap_subpixel_refinement,
        'weighted': decode_heatmap_weighted_average,
        'gaussian': decode_heatmap_gaussian_refinement,
    }
    
    if method not in decode_funcs:
        print(f'[Warning] Unknown method "{method}", using "argmax"')
        method = 'argmax'
    
    decode_func = decode_funcs[method]
    
    # 對每個關鍵點進行解碼
    kpts = []
    for k in range(K):
        hm = heatmap[k]
        x, y, score = decode_func(hm)
        
        # 縮放到原圖尺寸
        x = x * (orig_w / float(Wh))
        y = y * (orig_h / float(Hh))
        
        kpts.append((x, y, score))
    
    return kpts


def main():
    """
    主程式
    流程：解析參數 → 連接設備 → 載入韌體和模型 → 讀圖 → 推論 → 解碼 → 畫圖 → 存檔
    """
    # 設定命令列參數
    parser = argparse.ArgumentParser(
        description='HRNet pose demo on KL730 (GenericImageInference).\n'
                    '支援多種關鍵點偵測方式，可依需求選擇最適合的方法。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
偵測方法說明:
  argmax    - 簡單快速，直接取最大值位置 (預設，適合即時應用)
  subpixel  - 亞像素精度優化，使用梯度調整位置 (平衡速度與精度)
  weighted  - Top-K 加權平均，更穩定 (適合多雜訊場景)
  gaussian  - 高斯擬合優化，最精確但較慢 (適合高精度需求)
        '''
    )
    # ========================================================================
    # 自動找韌體和模型檔案的路徑
    # ========================================================================
    def _get_default_firmware_path():
        """
        自動找韌體檔案
        會依序檢查幾個可能的位置
        """
        # 優先：如果腳本在 plus_python 目錄中，直接使用相對路徑
        if os.path.basename(PWD) == 'plus_python':
            fw_path = os.path.join(PWD, 'res', 'firmware', 'KL730', 'kp_firmware.tar')
            fw_path = os.path.abspath(fw_path)  # 轉換為絕對路徑
            if os.path.exists(fw_path):
                return fw_path
        
        # 嘗試從 kneopi-examples-main 根目錄查找
        kneopi_root = _find_kneopi_examples_path()
        if kneopi_root:
            # 確保不會重複拼接路徑
            fw_path = os.path.join(kneopi_root, 'ai_application', 'plus_python', 'res', 'firmware', 'KL730', 'kp_firmware.tar')
            fw_path = os.path.abspath(fw_path)  # 轉換為絕對路徑
            if os.path.exists(fw_path):
                return fw_path
        
        # 如果腳本在專案根目錄，嘗試相對路徑
        fw_path = os.path.join(PWD, 'kneopi-examples-main', 'ai_application', 'plus_python', 'res', 'firmware', 'KL730', 'kp_firmware.tar')
        fw_path = os.path.abspath(fw_path)  # 轉換為絕對路徑
        if os.path.exists(fw_path):
            return fw_path
        
        # 返回預設路徑（即使不存在，讓後續錯誤處理來處理）
        return fw_path
    
    def _get_default_model_path():
        """
        自動找模型檔案
        會依序檢查幾個可能的位置（包括標準的 HRNet 模型路徑）
        """
        # 優先：檢查當前目錄
        model_path = os.path.join(PWD, 'models_730.nef')
        model_path = os.path.abspath(model_path)  # 轉換為絕對路徑
        if os.path.exists(model_path):
            return model_path
        
        # 如果腳本在 plus_python 目錄中，檢查標準模型路徑
        if os.path.basename(PWD) == 'plus_python':
            # 檢查 res/models/KL730/HRNet/models_730.nef（標準位置）
            standard_model = os.path.join(PWD, 'res', 'models', 'KL730', 'HRNet', 'models_730.nef')
            standard_model = os.path.abspath(standard_model)
            if os.path.exists(standard_model):
                return standard_model
            
            # 檢查 res/models/KL730/models_730.nef（備用位置）
            alt_model = os.path.join(PWD, 'res', 'models', 'KL730', 'models_730.nef')
            alt_model = os.path.abspath(alt_model)
            if os.path.exists(alt_model):
                return alt_model
            
            # 檢查祖父目錄（kneopi-examples-main 的父目錄，即專案根目錄）
            grandparent = os.path.dirname(os.path.dirname(PWD))  # 從 plus_python -> ai_application -> kneopi-examples-main
            root_dir = os.path.dirname(grandparent)  # kneopi-examples-main 的父目錄
            root_model = os.path.join(root_dir, 'models_730.nef')
            root_model = os.path.abspath(root_model)
            if os.path.exists(root_model):
                return root_model
            
            # 檢查祖父目錄本身（kneopi-examples-main）
            grandparent_model = os.path.join(grandparent, 'models_730.nef')
            grandparent_model = os.path.abspath(grandparent_model)
            if os.path.exists(grandparent_model):
                return grandparent_model
        
        # 嘗試從 kneopi-examples-main 根目錄查找標準模型路徑
        kneopi_root = _find_kneopi_examples_path()
        if kneopi_root:
            # 檢查標準模型路徑
            standard_model = os.path.join(kneopi_root, 'ai_application', 'plus_python', 'res', 'models', 'KL730', 'HRNet', 'models_730.nef')
            standard_model = os.path.abspath(standard_model)
            if os.path.exists(standard_model):
                return standard_model
            
            # 檢查備用模型路徑
            alt_model = os.path.join(kneopi_root, 'ai_application', 'plus_python', 'res', 'models', 'KL730', 'models_730.nef')
            alt_model = os.path.abspath(alt_model)
            if os.path.exists(alt_model):
                return alt_model
            
            # 檢查專案根目錄（kneopi-examples-main 的父目錄）
            root_model = os.path.join(os.path.dirname(kneopi_root), 'models_730.nef')
            root_model = os.path.abspath(root_model)
            if os.path.exists(root_model):
                return root_model
        
        # 返回預設路徑
        return model_path
    
    parser.add_argument('-p', '--port_id', type=int, default=0, help='USB port ID (default: 0)')
    parser.add_argument('-fw', '--firmware', 
                        default=_get_default_firmware_path(),
                        help='Path to kp_firmware.tar')
    parser.add_argument('-m', '--model', 
                        default=_get_default_model_path(),
                        help='Path to HRNet NEF')
    parser.add_argument('-img', '--image', default='people.jpg',
                        help='Path to input image')
    parser.add_argument('--thresh', type=float, default=0.2,
                        help='score threshold for drawing keypoints')
    parser.add_argument('--detect_method', type=str, default='argmax',
                        choices=['argmax', 'subpixel', 'weighted', 'gaussian'],
                        help='關鍵點偵測方法 (default: argmax)')
    parser.add_argument('--show_skeleton', action='store_true', default=True,
                        help='顯示骨架連線 (預設: 開啟)')
    parser.add_argument('--no_skeleton', dest='show_skeleton', action='store_false',
                        help='不顯示骨架連線')
    parser.add_argument('--show_labels', action='store_true', default=False,
                        help='顯示關節點名稱標籤 (預設: 關閉)')
    parser.add_argument('--show_scores', action='store_true', default=False,
                        help='顯示關節點分數 (預設: 關閉)')
    parser.add_argument('--point_size', type=int, default=5,
                        help='關節點大小 (預設: 5, 範圍: 3-10)')
    parser.add_argument('--line_thickness', type=int, default=3,
                        help='骨架線條粗細 (預設: 3, 範圍: 1-5)')
    parser.add_argument('--show_stats', action='store_true', default=True,
                        help='顯示統計資訊面板 (預設: 開啟)')
    parser.add_argument('--no_stats', dest='show_stats', action='store_false',
                        help='不顯示統計資訊面板')
    args = parser.parse_args()
    
    # 限制參數範圍（避免設定太極端的值）
    args.point_size = max(3, min(args.point_size, 10))
    args.line_thickness = max(1, min(args.line_thickness, 5))

    # ========================================================================
    # 檢查檔案路徑（韌體、模型、圖片）
    # ========================================================================
    # 如果已經是絕對路徑，直接使用；否則轉換為絕對路徑
    if os.path.isabs(args.firmware):
        firmware_path = args.firmware
    else:
        firmware_path = os.path.abspath(args.firmware)
    
    if os.path.isabs(args.model):
        model_path = args.model
    else:
        model_path = os.path.abspath(args.model)
    
    if os.path.isabs(args.image):
        image_path = args.image
    else:
        image_path = os.path.abspath(os.path.join(PWD, args.image))
    
    # 檢查韌體檔案是否存在
    if not os.path.exists(firmware_path):
        print(f'\033[91m[Error] Firmware file not found: {firmware_path}\033[0m')
        print(f'Please check the path or use -fw to specify the correct firmware path.')
        return
    
    # 檢查模型檔案是否存在
    if not os.path.exists(model_path):
        print(f'\033[91m[Error] Model file not found: {model_path}\033[0m')
        print(f'Please check the path or use -m to specify the correct model path.')
        return
    
    # 檢查圖片檔案是否存在
    if not os.path.exists(image_path):
        print(f'\033[91m[Error] Image file not found: {image_path}\033[0m')
        print(f'Please check the path or use -img to specify the correct image path.')
        return
    
    print(f'[File Check]')
    print(f'  Firmware: {firmware_path}')
    print(f'  Model: {model_path}')
    print(f'  Image: {image_path}')

    usb_port_id = args.port_id

    # ========================================================================
    # 連接設備
    # ========================================================================
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=usb_port_id)
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != usb_speed and kp.UsbSpeed.KP_USB_SPEED_HIGH != usb_speed:
            print('\033[93m[Warning] Device is not at super/high speed.\033[0m')
    except Exception as e:
        print(f'[Warning] USB speed check failed: {e}')

    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'\033[91m[Error] Connect device failed, port ID = \'{usb_port_id}\', error msg: [{str(exception)}]\033[0m')
        return

    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=10000)
    print(' - Success')

    # ========================================================================
    # 載入韌體和模型到設備
    # ========================================================================
    try:
        print('[Upload Firmware]')
        kp.core.load_firmware_from_file(device_group=device_group,
                                        scpu_fw_path=firmware_path,
                                        ncpu_fw_path='')
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'\033[91m[Error] Upload firmware failed, error = \'{str(exception)}\'\033[0m')
        print(f'Firmware path: {firmware_path}')
        return

    try:
        print('[Upload Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=model_path)
        print(' - Success')
    except kp.ApiKPException as exception:
        print(f'\033[91m[Error] Upload model failed, error = \'{str(exception)}\'\033[0m')
        print(f'Model path: {model_path}')
        return

    # ========================================================================
    # 讀取並預處理圖片
    # ========================================================================
    print('[Read Image]')
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f'\033[91m[Error] Cannot read image: {image_path}\033[0m')
        return

    orig_h, orig_w = img_bgr.shape[:2]
    in_h, in_w = _get_model_input_hw(model_nef_descriptor)
    print(f'[Model Input] HxW = {in_h}x{in_w}')

    # 縮放到模型輸入尺寸，然後轉成 RGB565 格式（設備需要的格式）
    resized = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
    img_bgr565 = cv2.cvtColor(resized, cv2.COLOR_BGR2BGR565)

    # ========================================================================
    # 準備推論輸入
    # ========================================================================
    generic_desc = kp.GenericImageInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
        input_node_image_list=[
            kp.GenericInputNodeImage(
                image=img_bgr565,
                resize_mode=kp.ResizeMode.KP_RESIZE_DISABLE,  # 已經在 CPU resize 好
                padding_mode=kp.PaddingMode.KP_PADDING_CORNER,
                normalize_mode=kp.NormalizeMode.KP_NORMALIZE_KNERON,
                image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565,
            )
        ]
    )

    # ========================================================================
    # 執行推論（送圖到設備，等結果回來）
    # ========================================================================
    print('[Inference]')
    kp.inference.generic_image_inference_send(
        device_group=device_group,
        generic_inference_input_descriptor=generic_desc
    )
    raw = kp.inference.generic_image_inference_receive(device_group=device_group)
    print(' - Success')

    # ========================================================================
    # 取出推論結果（heatmap）
    # ========================================================================
    # 只取第 0 個輸出節點（就是 heatmap）
    out0 = kp.inference.generic_inference_retrieve_float_node(
        node_idx=0,
        generic_raw_result=raw,
        channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW,
    )
    heatmap = _to_numpy(out0)  # 轉成 numpy array
    print(f'[Debug] heatmap shape: {heatmap.shape}')

    # ========================================================================
    # 解碼 heatmap 得到關節點，然後畫到圖片上
    # ========================================================================
    # 從 heatmap 解碼出 17 個關節點的座標
    print(f'[Detection Method] 使用 {args.detect_method} 方法')
    kpts = decode_heatmap_to_keypoints(
        heatmap, 
        orig_w=orig_w, 
        orig_h=orig_h,
        method=args.detect_method
    )
    
    # 畫關節點和骨架到圖片上
    print(f'[Visualization] Skeleton={args.show_skeleton}, Labels={args.show_labels}, Scores={args.show_scores}')
    vis = draw_keypoints_enhanced(
        img_bgr,
        kpts,
        threshold=args.thresh,
        show_skeleton=args.show_skeleton,
        show_labels=args.show_labels,
        show_scores=args.show_scores,
        point_radius=args.point_size,
        line_thickness=args.line_thickness
    )
    
    # ========================================================================
    # 計算統計資訊（顯示在終端和圖片上）
    # ========================================================================
    valid_count = sum(1 for (_, _, s) in kpts if s >= args.thresh)
    all_scores = [s for (_, _, s) in kpts]
    valid_scores = [s for (_, _, s) in kpts if s >= args.thresh]
    
    # 計算各種分數統計
    avg_score_all = sum(all_scores) / len(all_scores) if all_scores else 0.0
    avg_score_valid = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    max_score = max(all_scores) if all_scores else 0.0
    min_score_valid = min(valid_scores) if valid_scores else 0.0
    total_score = sum(valid_scores)
    
    # 在終端顯示統計資訊
    print(f'\n[Statistics]')
    print(f'  Valid Joints: {valid_count}/{len(kpts)}')
    print(f'  Average Score (All): {avg_score_all:.4f}')
    print(f'  Average Score (Valid): {avg_score_valid:.4f}')
    print(f'  Max Score: {max_score:.4f}')
    print(f'  Min Score (Valid): {min_score_valid:.4f}')
    print(f'  Total Score: {total_score:.4f}')
    
    # 在圖片左上角畫統計資訊面板（可選）
    if args.show_stats:
        # 根據圖片大小調整字體和間距
        img_h, img_w = vis.shape[:2]
        base_font_scale = 0.65 if img_w > 800 else 0.55
        line_height = int(28 * (img_w / 1000.0)) if img_w > 500 else 24
        line_height = max(22, min(line_height, 32))  # 限制在合理範圍
        padding = max(10, int(12 * (img_w / 1000.0)))
        
        # 優化資訊行（更簡潔易讀）
        info_lines = [
            f'Method: {args.detect_method}',
            f'Joints: {valid_count}/{len(kpts)} (Thr: {args.thresh:.2f})',
            f'Avg (All): {avg_score_all:.3f} | Avg (Valid): {avg_score_valid:.3f}',
            f'Max: {max_score:.3f} | Min: {min_score_valid:.3f}',
            f'Total: {total_score:.3f}'
        ]
        
        # 計算面板大小
        font_thickness = 1
        max_text_width = 0
        max_text_height = 0
        for line in info_lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, font_thickness)
            max_text_width = max(max_text_width, text_w)
            max_text_height = max(max_text_height, text_h + baseline)
        
        panel_width = max_text_width + padding * 2
        panel_height = len(info_lines) * line_height + padding * 2
        
        # 繪製帶圓角效果的背景面板（使用漸變透明）
        overlay = vis.copy()
        
        # 主要背景（深色半透明）
        cv2.rectangle(overlay, (padding, padding), (panel_width, panel_height), (20, 20, 20), -1)
        
        # 邊框（更柔和）
        cv2.rectangle(overlay, (padding, padding), (panel_width, panel_height), (100, 100, 100), 2)
        
        # 混合到原圖
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        
        # 繪製文字資訊（使用更柔和的顏色，統一字體大小）
        y_offset = padding + max_text_height + 5
        colors = [
            (100, 255, 100),    # 柔和綠色 - 方法
            (100, 255, 255),    # 柔和黃色 - 關節點數
            (200, 255, 255),    # 柔和青色 - 平均分數
            (200, 200, 255),    # 柔和藍色 - 最大最小
            (255, 200, 200)     # 柔和粉紅色 - 總分
        ]
        
        for i, line in enumerate(info_lines):
            color = colors[i] if i < len(colors) else (255, 255, 255)
            # 添加文字陰影效果（更清晰）
            cv2.putText(vis, line, (padding + 1, y_offset + 1),
                       cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
            cv2.putText(vis, line, (padding, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, base_font_scale, color, font_thickness, cv2.LINE_AA)
            y_offset += line_height

    # ========================================================================
    # 儲存結果圖片
    # ========================================================================
    out_path = 'hrnet_result.jpg'
    cv2.imwrite(out_path, vis)
    print(f'[Done] Saved: {out_path}')


if __name__ == '__main__':
    main()
