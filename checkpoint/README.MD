Tài liệu  hướng dẫn cách tích hợp 3 file Checkpoint (mô hình AI) vào hệ thống Backend.

Lưu ý: Tài liệu này giả định rằng Backend đã hoàn tất việc xử lý tín hiệu điện não đồ (EEG) và đã rút trích ra được một mảng (array) gồm 1214 đặc trưng (features). Nhiệm vụ của phần này là nhận 1214 đặc trưng đó và đưa ra kết quả dự đoán.

1. Giải phẫu 3 file Checkpoint (.pkl)

🧠 1. global_scaler.pkl (Bộ Chuẩn hóa Dữ liệu)

Nhiệm vụ: Khi  truyền mảng 1214 con số vào, file này sẽ chuẩn hóa (scale) các con số đó về cùng một hệ quy chiếu (mean = 0, std = 1) dựa trên tỷ lệ mà AI đã học từ 9 bệnh nhân. Nếu không chạy qua bộ này, mô hình sẽ bị "ngợp" bởi các con số quá lớn hoặc quá nhỏ và dự đoán sai lệch.

🔍 2. global_top_idx.pkl (Bộ Lọc Đặc trưng)
Bản chất: Là một mảng (Numpy Array) chứa chính xác 200 con số nguyên (ví dụ: [12, 45, 1102, ...]).

🤖 3. global_model_smart.pkl (Bộ Não Cốt lõi)
Bản chất: Là một mô hình học máy LGBMClassifier của thư viện LightGBM.

Nhiệm vụ: Nhận vào đúng 200 con số đã được làm sạch ở 2 bước trên, tính toán qua hàng ngàn cây quyết định bên trong và nhả ra một con số xác suất từ 0.0 đến 1.0 (Đại diện cho tỷ lệ rủi ro sắp xảy ra cơn động kinh).

2. Cài đặt Môi trường (Setup)
Trên Server Backend (Python), bạn cần cài đặt các thư viện sau để có thể đọc và chạy được 3 file .pkl:

        Bash
        pip install numpy scikit-learn lightgbm joblib
        Lưu ý: Bắt buộc phải có scikit-learn để đọc file scaler và lightgbm để đọc file model.

3. Luồng Code Tích hợp (Inference Pipeline)
Dưới đây là đoạn code mẫu (chuẩn xác nhất) để bạn nhúng vào API (Flask/FastAPI/Django).

Khuyến cáo: Việc load 3 file .pkl từ ổ cứng lên RAM tốn khoảng 1-2 giây. Để API chạy nhanh, hãy load 3 file này ra biến toàn cục (Global Variables) khi khởi động Server, KHÔNG load lại bên trong mỗi lượt gọi API.

----------------------------------------------------------------------
        Python
        import numpy as np
        import joblib

        # =====================================================================
        # BƯỚC A: LOAD CÁC FILE CHECKPOINT VÀO BỘ NHỚ RAM KHI KHỞI ĐỘNG SERVER
        # =====================================================================
        print("Đang nạp bộ nhớ AI...")
        SCALER = joblib.load('path/to/global_scaler.pkl')
        TOP_IDX = joblib.load('path/to/global_top_idx.pkl')
        MODEL = joblib.load('path/to/global_model_smart.pkl')
        print("Nạp AI thành công!")

        # =====================================================================
        # BƯỚC B: HÀM DỰ ĐOÁN (Gọi khi người dùng bấm nút phân tích trên Web)
        # =====================================================================
        def predict_seizure_from_features(features_1214):
            """
            Hàm nhận vào list 1214 đặc trưng và trả về kết quả chẩn đoán.
            
            Tham số:
            - features_1214: list hoặc numpy array 1D chứa 1214 con số (float).
            
            Trả về:
            - Dict chứa % rủi ro và cảnh báo.
            """
            
            # 1. Chuyển đổi đầu vào thành ma trận 2D (1 hàng, 1214 cột) chuẩn của Numpy
            X_input = np.array(features_1214, dtype=np.float32).reshape(1, -1)
            
            # 2. Bước 1: Chuẩn hóa dữ liệu với global_scaler
            X_scaled = SCALER.transform(X_input)
            
            # 3. Bước 2: Lọc lấy đúng 200 vị trí quan trọng nhất với global_top_idx
            X_final = X_scaled[:, TOP_IDX]
            
            # 4. Bước 3: Đưa vào global_model để lấy xác suất (Probability)
            # Lấy class 1 (cột thứ 2) - tỷ lệ bị động kinh
            probability = MODEL.predict_proba(X_final)[0, 1] 
            
            # 5. Phân tích kết quả dựa trên Ngưỡng (Threshold)
            # Ở đây set ngưỡng an toàn là 0.7 
            THRESHOLD = 0.7
            is_danger = bool(probability >= THRESHOLD)
            
            return {
                "risk_probability_percent": round(probability * 100, 2),
                "status": "Cảnh báo Tiền động kinh" if is_danger else "Bình thường",
                "is_danger": is_danger
            }

        # --- TEST THỬ CODE ---
        if __name__ == "__main__":
            # Giả lập mảng 1214 tính năng do phần xử lý tín hiệu truyền sang
            mock_features = np.random.rand(1214).tolist()
            
            result = predict_seizure_from_features(mock_features)
            print("Kết quả chẩn đoán:", result)







------------------------------------------------------------------------------
------------------------------------------------------------------------------
Hướng dẫn Tích hợp Module Trích xuất Đặc trưng (Feature Extractor)
Tài liệu này giải thích chi tiết cách sử dụng file feature_extractor.py. Đây là module cốt lõi làm nhiệm vụ xử lý tín hiệu điện não đồ (EEG) thô và biến đổi nó thành 1214 đặc trưng toán học để AI có thể hiểu được.

1. Nhiệm vụ của file feature_extractor.py

📥 Đầu vào - DATA đã processing từ EDF (Input): Một ma trận Numpy 2D kích thước 18 x 1280. Tương đương với 18 kênh điện cực, cắt trong khoảng thời gian đúng 5 giây (tần số lấy mẫu 256Hz).

📤 Đầu ra (Output): Một mảng (Numpy Array 1D) chứa chính xác 1214 con số (float).

2. Cài đặt thư viện
File này yêu cầu 2 thư viện toán học cơ bản của Python. Chạy lệnh sau trên Server:

        Bash
        pip install numpy scipy

3. Các hàm (Functions) quan trọng cần gọi
Bạn không cần phải hiểu hết các hàm tính toán phức tạp bên trong (như LBP, PSD, Hjorth...). Khi gọi vào API, bạn chỉ cần sử dụng 2 hàm duy nhất theo đúng thứ tự:

Hàm 1: preprocess_window(data, fs=256)
Tác dụng: Lọc nhiễu băng thông (0.5 - 40Hz), triệt nhiễu điện lưới (60Hz) và ép Z-score để chuẩn hóa biên độ sóng não.

Cách dùng: Truyền tín hiệu thô vào đây TRƯỚC KHI đem đi rút trích đặc trưng.

Hàm 2: extract_features(data, ch_names)
Tác dụng: Chạy toàn bộ luồng tính toán để đẻ ra 1214 con số.

Cách dùng: Yêu cầu truyền vào data (tín hiệu đã sạch từ hàm 1) và ch_names (danh sách tên 18 kênh).

4. Code Tích hợp Mẫu (Integration Template)
Dưới đây là luồng code hoàn chỉnh minh họa cách đọc file EDF và đưa qua feature_extractor.py. Bạn hãy nhúng logic này vào luồng API xử lý file tải lên của Backend.

        Python
        import numpy as np
        import mne
        # Nạp 2 hàm quan trọng từ file của Data Scientist
        from feature_extractor import preprocess_window, extract_features

        # =====================================================================
        # CẤU HÌNH BẮT BUỘC
        # Đảm bảo giữ đúng thứ tự 18 kênh này theo model đã huấn luyện
        # =====================================================================
        COMMON_CHANNELS = [
            'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
            'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
            'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
            'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
            'FZ-CZ', 'CZ-PZ'
        ]

        def process_edf_to_features(edf_file_path):
            """
            Hàm đọc file EDF, lấy 5 giây đầu và trích xuất ra 1214 features.
            """
            # ---------------------------------------------------------
            # BƯỚC 1: ĐỌC FILE EDF (Dùng thư viện mne)
            # ---------------------------------------------------------
            raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
            
            # Chuẩn hóa về tần số 256Hz nếu file gốc khác tần số
            if raw.info['sfreq'] != 256.0:
                raw.resample(256.0)
                
            # Lọc lấy đúng 18 kênh cần thiết
            raw.pick_channels(COMMON_CHANNELS, ordered=True)
            
            # Cắt lấy 5 giây đầu tiên (từ sample 0 đến 1280)
            # Nếu muốn cắt ở giây khác, thay đổi giá trị index này.
            data, times = raw[:, 0:1280] 
            
            # Đổi đơn vị Volts của mne sang Microvolts để khớp với Model
            raw_matrix = (data * 1e6).astype(np.float32)
            
            # ---------------------------------------------------------
            # BƯỚC 2: TIỀN XỬ LÝ (PREPROCESSING)
            # ---------------------------------------------------------
            print("Đang làm sạch và chuẩn hóa tín hiệu Z-score...")
            clean_matrix = preprocess_window(raw_matrix, fs=256)
            
            # ---------------------------------------------------------
            # BƯỚC 3: TRÍCH XUẤT 1214 ĐẶC TRƯNG
            # ---------------------------------------------------------
            print("Đang chạy thuật toán LBP, PSD, Stats, Connectivity...")
            features_1214 = extract_features(clean_matrix, COMMON_CHANNELS)
            
            # Trả về list gồm 1214 phần tử để đưa tiếp vào Model AI (.pkl)
            return features_1214.tolist()

        # --- TEST THỬ ---
        if __name__ == "__main__":
            # Thay bằng đường dẫn file EDF thực tế của bạn
            # features = process_edf_to_features("test_patient.edf")
            # print(f"Đã trích xuất thành công: {len(features)} đặc trưng.")
            pass
