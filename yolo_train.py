from ultralytics import YOLO
import torch
from pathlib import Path
import logging
from datetime import datetime

# Настройка логирования
def setup_logger():
    logger = logging.getLogger('YOLO_Training')
    logger.setLevel(logging.INFO)
    
    # Создаем директорию для логов
    log_dir = Path('runs/train/logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Хендлер для файла
    file_handler = logging.FileHandler(
        log_dir / f'train_{datetime.now():%Y%m%d_%H%M%S}.log'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Хендлер для консоли
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def train_yolo():
    logger = setup_logger()
    logger.info("Начало тренировки YOLO11")
    
    try:
        # Инициализация модели
        model = YOLO('yolo11l.pt')  # или другая версия YOLO11
        
        # Параметры тренировки
        train_args = {
            'data': 'dataset_work/dataset.yaml',
            'epochs': 200,  # Максимальное количество эпох
            'patience': 10,  # Количество эпох без улучшений для ранней остановки
            'batch': 16,
            'imgsz': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 8,
            'project': 'runs/train',
            'name': f'exp_{datetime.now():%Y%m%d_%H%M%S}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'auto',
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,
            'fraction': 1.0,
            'cache': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'save': True,
            'save_json': True,
            'save_hybrid': False,
            'conf': 0.001,
            'iou': 0.6,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True
        }
        
        # Запуск тренировки
        logger.info("Запуск тренировки...")
        results = model.train(**train_args)
        
        # Анализ результатов
        logger.info("Тренировка завершена")
        # Получение результатов валидации
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            logger.info(f"Лучшая модель сохранена в: {best_model_path}")
            
            # Валидация лучшей модели для получения точных метрик
            best_model = YOLO(best_model_path)
            val_results = best_model.val(data='dataset_work/dataset.yaml')
            
            # Получение метрик
            logger.info(f"mAP50-95: {val_results.box.map}")
            logger.info(f"mAP50: {val_results.box.map50}")
            logger.info(f"Precision: {val_results.box.p}")
            logger.info(f"Recall: {val_results.box.r}")
            
            # Сохранение результатов
            logger.info(f"Полные результаты сохранены в: {results.save_dir}")
        
    except Exception as e:
        logger.error(f"Ошибка во время тренировки: {str(e)}")
        raise

if __name__ == "__main__":
    train_yolo()