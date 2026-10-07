import pandas as pd
import os
import shutil

# 1. Cấu hình đường dẫn và nhãn cần lấy
ORIGINAL_CSV = 'styles.csv'
ORIGINAL_IMG_DIR = 'images'
NEW_CSV = 'clean_dataset.csv'
NEW_IMG_DIR = 'training_images'

# Chọn 4 loại đồ khác biệt rõ ràng nhất để phân loại
TARGET_CATEGORIES = ['Tshirts', 'Casual Shoes', 'Watches', 'Sunglasses']
SAMPLES_PER_CATEGORY = 1000

# 2. Tạo thư mục mới để chứa ảnh sạch
if not os.path.exists(NEW_IMG_DIR):
    os.makedirs(NEW_IMG_DIR)

print("Đang đọc file CSV...")
# Bỏ qua các dòng lỗi cú pháp trong file CSV gốc bằng on_bad_lines='skip'
df = pd.read_csv(ORIGINAL_CSV, on_bad_lines='skip')

# Lọc chỉ lấy 4 loại đồ đã chọn
df_filtered = df[df['articleType'].isin(TARGET_CATEGORIES)]

valid_rows = []

print("Đang lọc ảnh và copy dữ liệu")
# 3. Lấy ngẫu nhiên và copy ảnh
for category in TARGET_CATEGORIES:
    category_df = df_filtered[df_filtered['articleType'] == category]
    
    count = 0
    # Xáo trộn ngẫu nhiên và lặp qua từng dòng
    for index, row in category_df.sample(frac=1, random_state=42).iterrows():
        if count >= SAMPLES_PER_CATEGORY:
            break
            
        img_filename = f"{row['id']}.jpg"
        orig_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)
        new_img_path = os.path.join(NEW_IMG_DIR, img_filename)
        
        # Chỉ copy nếu file ảnh thực sự tồn tại
        if os.path.exists(orig_img_path):
            shutil.copy2(orig_img_path, new_img_path)
            valid_rows.append(row)
            count += 1
            
    print(f"Đã copy {count} ảnh cho loại: {category}")

# 4. Lưu danh sách mới ra file CSV
new_df = pd.DataFrame(valid_rows)
new_df.to_csv(NEW_CSV, index=False)

print(f"\nHoàn tất! Tổng số ảnh hợp lệ: {len(new_df)}")
print(f"File nhãn mới đã lưu tại: {NEW_CSV}")
print(f"Ảnh đã được copy sang thư mục: {NEW_IMG_DIR}")