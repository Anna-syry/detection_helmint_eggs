import json
import os
from pathlib import Path

def convert_normalized_to_absolute(annotations_path):
    # Загружаем JSON файл
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Создаем словарь изображений для быстрого доступа
    images_dict = {img['id']: img for img in data['images']}
    
    # Конвертируем координаты
    for ann in data['annotations']:
        img = images_dict[ann['image_id']]
        width = img['width']
        height = img['height']
        
        # Конвертируем нормализованные координаты в абсолютные
        x, y, w, h = ann['bbox']
        ann['bbox'] = [
            x * width,  # x
            y * height,  # y
            w * width,  # width
            h * height  # height
        ]
        # Обновляем площадь
        ann['area'] = (w * width) * (h * height)
    
    # Сохраняем обновленный JSON
    output_path = str(Path(annotations_path).parent / 'annotations_absolute.json')
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_path

if __name__ == '__main__':
    # Конвертируем тренировочные аннотации
    train_annotations = '/home/dataset_work/train/annotations.json'
    val_annotations = '/home/dataset_work/val/annotations.json'
    test_annotations = '/home/dataset_work/test/annotations.json'
    
    train_output = convert_normalized_to_absolute(train_annotations)
    val_output = convert_normalized_to_absolute(val_annotations)
    test_output = convert_normalized_to_absolute(test_annotations)
    
    print(f"Converted annotations saved to:\n{train_output}\n{val_output}\n{test_output}") 