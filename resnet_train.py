import os
import torch
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode
from detectron2.data.datasets import register_coco_instances
from detectron2.model_zoo import model_zoo
from detectron2.evaluation import COCOEvaluator
import logging
from detectron2.utils.events import get_event_storage, EventStorage
import argparse
import numpy as np
import csv
import time
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Настраиваем логгирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("detectron2")

# Регистрация датасетов
register_coco_instances(
    "eggs_dataset_train", 
    {}, 
    "../dataset_work/train/annotations_absolute.json",
    "../dataset_work/train/images"
)

register_coco_instances(
    "eggs_dataset_val", 
    {},
    "../dataset_work/val/annotations_absolute.json",
    "../dataset_work/val/images"
)

class MetricsLogger:
    """Класс для логирования метрик в CSV-файл."""
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.csv_path = self.output_dir / f"metrics_{datetime.now():%Y%m%d_%H%M%S}.csv"
        self.metrics_history = []
        self.header = ["iteration", "time", "loss_total", "loss_rpn_cls", "loss_rpn_loc", 
                      "loss_cls", "loss_box_reg", "lr", "is_validation",
                      "val_AP", "val_AP50", "val_AP75", "val_APs", "val_APm", "val_APl"]
        
        # Создаем директорию, если она не существует
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем CSV-файл с заголовками
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        
        logger.info(f"Метрики будут сохраняться в: {self.csv_path}")
        
        # Время начала тренировки
        self.start_time = time.time()
    
    def update(self, iteration, losses, lr, validation_results=None):
        """Обновляет историю метрик и записывает в CSV."""
        elapsed_time = time.time() - self.start_time
        
        # Собираем метрики для текущей итерации
        metrics = {
            "iteration": iteration,
            "time": f"{elapsed_time:.1f}",
            "lr": f"{lr:.6f}"
        }
        
        # Добавляем общую потерю, если есть потери
        if losses and len(losses) > 0:
            metrics["loss_total"] = f"{sum(losses.values()):.4f}"
        else:
            metrics["loss_total"] = "N/A"
        
        # Добавляем отдельные потери, если они доступны
        if losses:
            for k, v in losses.items():
                metrics[k] = f"{v:.4f}"
        
        # Добавляем метрики валидации, если они доступны
        if validation_results and "bbox" in validation_results:
            for k, v in validation_results["bbox"].items():
                # Преобразуем nan в 0 для корректного отображения в CSV
                if isinstance(v, float) and np.isnan(v):
                    metrics[f"val_{k}"] = "0.0000"
                else:
                    metrics[f"val_{k}"] = f"{v:.4f}"
            
            # Добавляем флаг, что это итерация с валидацией
            metrics["is_validation"] = "True"
        else:
            # Для строк без валидации добавляем пустые значения
            metrics["is_validation"] = "False"
        
        # Сохраняем в историю
        self.metrics_history.append(metrics)
        
        # Записываем в CSV
        row = []
        for field in self.header:
            row.append(metrics.get(field, "N/A"))
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        return metrics

class EarlyStoppingTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
    
    def __init__(self, cfg):
        super().__init__(cfg)
        # Параметры раннего останова
        self.patience = cfg.SOLVER.EARLY_STOPPING_PATIENCE
        self.min_delta = cfg.SOLVER.EARLY_STOPPING_MIN_DELTA
        self.metric = cfg.SOLVER.EARLY_STOPPING_METRIC
        
        # Отслеживание лучшей метрики
        self.best_metric = -float('inf')
        self.wait_count = 0
        self.should_stop = False
        
        # Инициализация логгера метрик
        self.metrics_logger = MetricsLogger(cfg.OUTPUT_DIR)
        
        logger.info(f"Ранний останов настроен: терпение={self.patience}, мин_дельта={self.min_delta}, метрика={self.metric}")
    
    def after_step(self):
        """Записывает дополнительные метрики после каждого шага и проверяет условие раннего останова."""
        # Вызываем базовый метод
        super().after_step()
        
        # Записываем текущие потери для наглядности
        storage = get_event_storage()
        
        # Получаем текущие потери и скорость обучения
        losses = {}
        for k, v in storage.histories().items():
            if "loss" in k:
                # Проверяем, есть ли значения в буфере истории, используя метод latest()
                try:
                    latest_value = v.latest()
                    losses[k] = latest_value
                except IndexError:
                    # Если буфер пуст, пропускаем
                    pass
        
        # Получаем текущую скорость обучения
        lr = self.optimizer.param_groups[0]["lr"]
        
        # Логируем метрики каждые 20 итераций
        if self.iter % 20 == 0:
            self.metrics_logger.update(self.iter, losses, lr)
        
        # Каждые N итераций делаем оценку вручную
        if (self.iter + 1) % self.cfg.TEST.EVAL_PERIOD == 0:
            logger.info(f"\n=== Оценка на итерации {self.iter + 1} ===")
            results = self.test(self.cfg, self.model)
            
            # Явно печатаем результаты для отладки
            logger.info(f"Результаты оценки: {results}")
            
            # Логируем метрики валидации
            self.metrics_logger.update(self.iter, losses, lr, results)
            
            # Записываем метрики в TensorBoard
            if results:
                # Записываем основную метрику AP
                if "bbox" in results:
                    for k, v in results["bbox"].items():
                        storage.put_scalar(f"validation/{k}", v)
                    
                    # Особо отмечаем важные метрики
                    if "AP" in results["bbox"]:
                        storage.put_scalar("mAP", results["bbox"]["AP"])
                    if "AP50" in results["bbox"]:
                        storage.put_scalar("AP50", results["bbox"]["AP50"])
                    
                    # Проверяем условие раннего останова
                    self._check_early_stopping(results)
    
    def _check_early_stopping(self, results):
        """Проверяет условие раннего останова на основе результатов валидации."""
        if "bbox" not in results or self.metric not in results["bbox"]:
            logger.warning(f"Метрика {self.metric} не найдена в результатах оценки")
            return
        
        current_metric = results["bbox"][self.metric]
        logger.info(f"Текущее значение метрики {self.metric}: {current_metric}")
        logger.info(f"Лучшее значение метрики до сих пор: {self.best_metric}")
        
        if current_metric > self.best_metric + self.min_delta:
            # Метрика улучшилась
            logger.info(f"Метрика улучшилась с {self.best_metric} до {current_metric}")
            self.best_metric = current_metric
            self.wait_count = 0
            
            # Сохраняем лучшую модель отдельно
            self.checkpointer.save(f"model_best_{self.metric}")
        else:
            # Метрика не улучшилась
            self.wait_count += 1
            logger.info(f"Метрика не улучшилась. Счетчик ожидания: {self.wait_count}/{self.patience}")
            
            if self.wait_count >= self.patience:
                logger.info(f"Ранний останов на итерации {self.iter + 1}! Метрика {self.metric} не улучшалась в течение {self.patience} проверок.")
                self.should_stop = True
    
    def train(self):
        """Переопределяем метод train для поддержки раннего останова."""
        self.start_iter = self.iter
        self.max_iter = self.cfg.SOLVER.MAX_ITER
        
        with EventStorage(self.start_iter) as storage:
            self.storage = storage
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
                    
                    # Проверяем условие раннего останова
                    if self.should_stop:
                        logger.info("Обучение остановлено по критерию раннего останова!")
                        break
                        
            except Exception as e:
                logger.exception("Исключение во время обучения:")
                raise
            finally:
                self.after_train()
        
        # Загружаем лучшую модель в конце обучения
        best_model_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_best_{self.metric}.pth")
        if os.path.exists(best_model_path):
            logger.info(f"Загрузка лучшей модели из {best_model_path}")
            self.checkpointer.load(best_model_path)
        
        return self.test(self.cfg, self.model)

def add_early_stopping_config(cfg):
    """Добавляет параметры раннего останова в конфигурацию."""
    # Добавляем параметры раннего останова в существующий раздел SOLVER
    cfg.SOLVER.EARLY_STOPPING_PATIENCE = 5  # Количество проверок без улучшения
    cfg.SOLVER.EARLY_STOPPING_MIN_DELTA = 0.001  # Минимальное улучшение метрики
    cfg.SOLVER.EARLY_STOPPING_METRIC = "AP"  # Метрика для отслеживания (AP, AP50, AP75 и т.д.)
    return cfg

def setup_cfg(model_type='faster_rcnn_r50'):
    """
    Настройка конфигурации в зависимости от выбранной модели.
    
    Args:
        model_type: Тип модели ('faster_rcnn_r50', 'faster_rcnn_r101', 'faster_rcnn_x101')
    """
    cfg = get_cfg()
    
    # Словарь с конфигурациями для разных моделей
    model_configs = {
        'faster_rcnn_r50': {
            'config_file': "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
            'output_dir': "output/egg_detection_faster_r50"
        },
        'faster_rcnn_r101': {
            'config_file': "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
            'output_dir': "output/egg_detection_faster_r101"
        },
        'faster_rcnn_x101': {
            'config_file': "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
            'output_dir': "output/egg_detection_faster_x101"
        }
    }
    
    # Проверяем, что выбранная модель существует в словаре
    if model_type not in model_configs:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные варианты: {list(model_configs.keys())}")
    
    # Получаем конфигурацию для выбранной модели
    selected_config = model_configs[model_type]
    config_file = selected_config['config_file']
    output_dir = selected_config['output_dir']
    
    # Загружаем базовую конфигурацию из Model Zoo
    logger.info(f"Загрузка конфигурации: {config_file}")
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    # Указываем путь для предобученной модели
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    
    # Настройка путей к датасетам
    cfg.DATASETS.TRAIN = ("eggs_dataset_train",)
    cfg.DATASETS.TEST = ("eggs_dataset_val",)
    
    # Основные параметры обучения для датасета яиц
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (2000, 3000)
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.IMS_PER_BATCH = 2
    
    # Устанавливаем количество классов (2 класса яиц)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    # Настраиваем периодичность оценки и сохранения
    cfg.TEST.EVAL_PERIOD = 200
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    
    # Использование FP16 для ускорения
    cfg.SOLVER.AMP.ENABLED = True
    
    # Вывод дополнительной информации
    cfg.DATALOADER.NUM_WORKERS = 2
    
    # Директория для выходных данных
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Добавляем параметры раннего останова
    cfg = add_early_stopping_config(cfg)
    
    return cfg

def main():
    # Добавляем аргумент командной строки для выбора модели
    parser = argparse.ArgumentParser(description='Обучение детектора яиц')
    parser.add_argument('--model', type=str, default='faster_rcnn_r50',
                        choices=['faster_rcnn_r50', 'faster_rcnn_r101', 'faster_rcnn_x101'],
                        help='Тип модели для обучения')
    parser.add_argument('--patience', type=int, default=5,
                        help='Количество проверок без улучшения для раннего останова')
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Минимальное улучшение метрики')
    parser.add_argument('--metric', type=str, default='AP',
                        choices=['AP', 'AP50', 'AP75'],
                        help='Метрика для отслеживания раннего останова')
    args = parser.parse_args()
    
    logger.info(f"Выбрана модель: {args.model}")
    logger.info("Настройка конфигурации...")
    cfg = setup_cfg(model_type=args.model)
    
    # Обновляем параметры раннего останова из аргументов командной строки
    cfg.SOLVER.EARLY_STOPPING_PATIENCE = args.patience
    cfg.SOLVER.EARLY_STOPPING_MIN_DELTA = args.min_delta
    cfg.SOLVER.EARLY_STOPPING_METRIC = args.metric
    
    logger.info("Создание тренера...")
    trainer = EarlyStoppingTrainer(cfg)
    
    logger.info("Загрузка предобученных весов...")
    trainer.resume_or_load(resume=False)
    
    logger.info("Начало обучения...")
    final_results = trainer.train()
    
    logger.info(f"Обучение завершено. Финальные результаты: {final_results}")

if __name__ == "__main__":
    main() 