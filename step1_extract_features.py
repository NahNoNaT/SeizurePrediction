import os
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import MinMaxScaler
import warnings
import concurrent.futures
from tqdm import tqdm
import gc

# 1. HÀM TRÍCH XUẤT (GIỮ NGUYÊN BẢN ĐÃ FIX n_bins=10)
def extract_glcm_lbp_separated(spectrogram_2d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scaler = MinMaxScaler(feature_range=(0, 255))
        img_gray = scaler.fit_transform(spectrogram_2d).astype(np.uint8)

    # ĐỔI distances=[1, 2] THÀNH distances=[1]
    glcm = graycomatrix(img_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    contrast      = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity   = graycoprops(glcm, 'homogeneity').flatten()
    energy        = graycoprops(glcm, 'energy').flatten()
    correlation   = graycoprops(glcm, 'correlation').flatten()
    glcm_vector = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation]).astype(np.float32)

    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    
    # Đã fix lỗi 10 chiều
    n_bins = n_points + 2 
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    lbp_vector = lbp_hist.astype(np.float32)

    return glcm_vector, lbp_vector

def process_single_sample(sample_3d):
    sample_glcm, sample_lbp = [], []
    for ch in range(sample_3d.shape[0]):
        g_vec, l_vec = extract_glcm_lbp_separated(sample_3d[ch])
        sample_glcm.append(g_vec)
        sample_lbp.append(l_vec)
    return np.concatenate(sample_glcm), np.concatenate(sample_lbp)

# ==========================================
# 2. HÀM XỬ LÝ AN TOÀN CHỐNG SẬP RAM
# ==========================================
def extract_and_save_features_safe(data_dir, glcm_dir, lbp_dir):
    os.makedirs(glcm_dir, exist_ok=True)
    os.makedirs(lbp_dir, exist_ok=True)
    
    patients = ['chb01','chb02','chb03','chb05','chb09','chb10',
                'chb13','chb14','chb18','chb19','chb20','chb21','chb23']
    
    # ⚠️ GIỚI HẠN AN TOÀN: Chỉ cho phép 4 tiến trình (Máy vẫn mượt, RAM không nổ)
    MAX_WORKERS = 8 
    
    for pt in patients:
        npz_file = os.path.join(data_dir, f"{pt}.npz")
        
        # Bỏ qua nếu đã xong
        if os.path.exists(os.path.join(glcm_dir, f"{pt}_glcm.npz")) and \
           os.path.exists(os.path.join(lbp_dir, f"{pt}_lbp.npz")):
            print(f"\n⏭️ Đã xong {pt.upper()} từ trước, bỏ qua.")
            continue
            
        if not os.path.exists(npz_file):
            continue
            
        print(f"\n🚀 Đang xử lý: {pt.upper()} (Tốc độ an toàn: {MAX_WORKERS} luồng)")
        
        try:
            with np.load(npz_file, allow_pickle=True, mmap_mode='r') as data:
                
                # Hàm helper nạp data lắt nhắt
                def run_parallel_safe(data_mmap, desc):
                    num_samples = data_mmap.shape[0]
                    res_g, res_l = [], []
                    
                    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        # Dùng chunksize=10: Nghĩa là phát cho mỗi nhân 10 bức ảnh để làm, 
                        # làm xong mới phát tiếp. Không nhồi toàn bộ vào RAM cùng lúc!
                        # Sử dụng generator (i for i...) để tiết kiệm bộ nhớ
                        futures = executor.map(process_single_sample, 
                                               (data_mmap[i] for i in range(num_samples)), 
                                               chunksize=10)
                        
                        for g, l in tqdm(futures, total=num_samples, desc=desc, leave=False):
                            res_g.append(g)
                            res_l.append(l)
                            
                    return np.array(res_g), np.array(res_l)

                pre_glcm, pre_lbp = run_parallel_safe(data['preictal'], "  -> Preictal  ")
                inter_glcm, inter_lbp = run_parallel_safe(data['interictal'], "  -> Interictal")

                # Lưu (Dùng save thay vì savez_compressed để tránh Spike RAM lúc lưu)
                glcm_out = os.path.join(glcm_dir, f"{pt}_glcm_temp.npz")
                np.savez(glcm_out, 
                         preictal=pre_glcm, interictal=inter_glcm,
                         preictal_seizure_ids=data['preictal_seizure_ids'],
                         interictal_group_ids=data['interictal_group_ids'],
                         n_folds=data['n_folds'])
                os.rename(glcm_out, os.path.join(glcm_dir, f"{pt}_glcm.npz"))
                
                lbp_out = os.path.join(lbp_dir, f"{pt}_lbp_temp.npz")
                np.savez(lbp_out, 
                         preictal=pre_lbp, interictal=inter_lbp,
                         preictal_seizure_ids=data['preictal_seizure_ids'],
                         interictal_group_ids=data['interictal_group_ids'],
                         n_folds=data['n_folds'])
                os.rename(lbp_out, os.path.join(lbp_dir, f"{pt}_lbp.npz"))
                
                print(f"  ✅ Đã lưu an toàn: {pt}")
                
        except Exception as e:
            print(f"  ❌ Lỗi ở {pt}: {e}")
        finally:
            # Ép HĐH thu hồi RAM ngay lập tức
            gc.collect() 

if __name__ == "__main__":
    DATA_DIR = r"D:/archive/processed_data(5)" 
    GLCM_DIR = r"D:/archive/features_GLCM"
    LBP_DIR = r"D:/archive/features_LBP"
    extract_and_save_features_safe(DATA_DIR, GLCM_DIR, LBP_DIR)