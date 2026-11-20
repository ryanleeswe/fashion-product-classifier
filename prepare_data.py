import pandas as pd
import os
import shutil

# Configuration paths
ORIGINAL_CSV = 'styles.csv'
ORIGINAL_IMG_DIR = 'images'
NEW_CSV = 'clean_dataset.csv'
NEW_IMG_DIR = 'training_images'

# Eight classes used by clean_dataset.csv (1,000 images per class).
TARGET_CATEGORIES = [
    'Casual Shoes', 'Handbags', 'Shirts', 'Sports Shoes',
    'Watches', 'Tops', 'Tshirts', 'Jeans',
]
SAMPLES_PER_CATEGORY = 1000

# Create output image directory.
if not os.path.exists(NEW_IMG_DIR):
    os.makedirs(NEW_IMG_DIR)

print("Reading source CSV...")
df = pd.read_csv(ORIGINAL_CSV, on_bad_lines='skip')

# Keep the target classes.
df_filtered = df[df['articleType'].isin(TARGET_CATEGORIES)]
valid_rows = []

print("Filtering and copying images...")
for category in TARGET_CATEGORIES:
    category_df = df_filtered[df_filtered['articleType'] == category]
    count = 0
    for _, row in category_df.sample(frac=1, random_state=42).iterrows():
        if count >= SAMPLES_PER_CATEGORY:
            break
        img_filename = f"{row['id']}.jpg"
        orig_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)
        new_img_path = os.path.join(NEW_IMG_DIR, img_filename)
        if os.path.exists(orig_img_path):
            shutil.copy2(orig_img_path, new_img_path)
            valid_rows.append(row)
            count += 1

    if count != SAMPLES_PER_CATEGORY:
        raise ValueError(
            f"Category {category!r} has only {count}/{SAMPLES_PER_CATEGORY} valid images."
        )
    print(f"Copied {count} images for: {category}")

new_df = pd.DataFrame(valid_rows)
new_df.to_csv(NEW_CSV, index=False)
print(f"Done. Wrote {len(new_df)} rows to {NEW_CSV}")
print(f"Images copied to {NEW_IMG_DIR}")
