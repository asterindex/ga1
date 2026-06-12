"""
Завантаження Military and Civilian Vehicles Classification датасету
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from src import config
from typing import Tuple


# Mapping класів до індексів
CLASS_MAPPING = {
    'civilian aircraft': 0,
    'civilian car': 1,
    'military aircraft': 2,
    'military helicopter': 3,
    'military tank': 4,
    'military truck': 5
}

CLASS_NAMES = list(CLASS_MAPPING.keys())


def load_dataset(subset_size: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантажує Military and Civilian Vehicles Classification датасет
    
    Dataset structure:
    - data/Images/ - всі зображення
    - data/Labels/CSV Format/train_labels.csv - train анотації (bounding boxes)
    - data/Labels/CSV Format/test_labels.csv - test анотації
    
    Args:
        subset_size: Якщо вказано, використовувати лише цю кількість зразків (для fast mode)
    
    Returns: X_train, y_train, X_test, y_test
    """
    print(f"\n  📊 ЗАВАНТАЖЕННЯ MILITARY VEHICLES ДАТАСЕТУ")
    print("  " + "="*60)
    
    # Шляхи до даних
    images_dir = os.path.join(config.DATASET_PATH, 'Images')
    train_csv = os.path.join(config.DATASET_PATH, 'Labels', 'CSV Format', 'train_labels.csv')
    test_csv = os.path.join(config.DATASET_PATH, 'Labels', 'CSV Format', 'test_labels.csv')
    
    # Завантаження анотацій
    print(f"    📂 Завантаження анотацій...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"    📊 Train: {len(train_df)} об'єктів")
    print(f"    📊 Test:  {len(test_df)} об'єктів")
    
    # Завантаження зображень
    print(f"    🖼️  Завантаження зображень (resize до {config.IMAGE_SIZE}x{config.IMAGE_SIZE})...")
    
    X_train, y_train = _load_images_from_csv(train_df, images_dir, subset_size)
    X_test, y_test = _load_images_from_csv(test_df, images_dir, subset_size // 5 if subset_size else None)
    
    print(f"    ✅ Train: {X_train.shape[0]:,} зображень, shape: {X_train.shape}")
    print(f"    ✅ Test:  {X_test.shape[0]:,} зображень, shape: {X_test.shape}")
    
    # Розподіл класів
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print(f"\n    📊 Розподіл класів:")
    
    for cls, count in zip(unique_train, counts_train):
        class_name = CLASS_NAMES[cls]
        print(f"       Клас {cls} ({class_name:20s}): {count:5d} ({count/len(y_train)*100:5.1f}%)")
    
    print(f"    📊 Всього класів: {len(unique_train)}")
    print("  " + "="*60 + "\n")
    
    return X_train, y_train, X_test, y_test


def _load_images_from_csv(df: pd.DataFrame, images_dir: str, limit: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Завантажує зображення з CSV файлу
    
    CSV format: filename,width,height,class,xmin,ymin,xmax,ymax
    Кожен рядок - це bounding box одного об'єкта.
    
    Для класифікації беремо цілі зображення (не crop bounding boxes),
    і присвоюємо клас першого об'єкта на зображенні.
    """
    # Групуємо по filename - беремо перший об'єкт на кожному зображенні
    df_grouped = df.groupby('filename').first().reset_index()
    
    if limit:
        df_grouped = df_grouped[:limit]
    
    X_list = []
    y_list = []
    
    skipped = 0
    for idx, row in df_grouped.iterrows():
        try:
            # Завантаження зображення
            img_path = os.path.join(images_dir, row['filename'])
            
            if not os.path.exists(img_path):
                skipped += 1
                continue
            
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BILINEAR)
            
            # Конвертація в numpy array та нормалізація
            img_array = np.array(img, dtype='float32') / 255.0
            
            # Клас
            class_name = row['class']
            if class_name not in CLASS_MAPPING:
                skipped += 1
                continue
            
            class_id = CLASS_MAPPING[class_name]
            
            X_list.append(img_array)
            y_list.append(class_id)
            
        except Exception as e:
            if skipped < 5:  # Show only first 5 errors
                print(f"    ⚠️  Помилка завантаження {row['filename']}: {e}")
            skipped += 1
            continue
    
    if skipped > 0:
        print(f"    ⚠️  Пропущено {skipped} зображень")
    
    X = np.array(X_list, dtype='float32')
    y = np.array(y_list, dtype='int32')
    
    return X, y


def get_dataset_info() -> Tuple[tuple, int]:
    """Return input_shape and output_dim for dataset"""
    input_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    output_dim = config.NUM_CLASSES
    return input_shape, output_dim
