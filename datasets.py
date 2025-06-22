"""
Полный класс подготовки датасета яиц гельминтов для компьютерного зрения
с поддержкой современных моделей глубокого обучения
"""

import os
import json
import shutil
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, OrderedDict
from tqdm import tqdm
import albumentations as A
from pycocotools.coco import COCO
import fiftyone as fo
from roboflow import Roboflow
import yaml
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
from datetime import datetime


# Установите переменную окружения в Python
os.environ["ROBOFLOW_API_KEY"] = "gTEze7gCygGP5tlKAK2a"


class HelminthDatasetPreprocessor:
    """
    Полный цикл подготовки датасета для медицинских изображений с поддержкой:
    - Интеграции с Roboflow
    - Стратифицированного разделения данных
    - Расширенной аугментации
    - Поддержки форматов YOLO, COCO
    - Визуализации и валидации данных

    Пример использования:
    ```
    preprocessor = HelminthDatasetPreprocessor(
        roboflow_workspace="helminth-project",
        roboflow_project="egg-detection",
        class_config={"egg": ["ascaris", "taenia"]},
        target_formats=["yolo", "coco"]
    )
    preprocessor.prepare_dataset()
    preprocessor.visualize_sample()
    ```
    """

    def __init__(
        self,
        dataset_src: str = "dataset_src",
        dataset_work: str = "dataset_work",
        roboflow_workspace: Optional[str] = None,
        roboflow_project: Optional[str] = None,
        roboflow_version: int = 1,
        force_download: bool = False,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        augmentation_config: Optional[Dict] = None,
        class_config: Optional[Dict[str, List[str]]] = None,
        target_formats: List[str] = ["yolo", "coco"],
        seed: int = 42,
        run_id: Optional[int] = None,
        save_dir: str = "runs",
        max_samples: Optional[int] = None
    ):
        self.dataset_src = dataset_src
        self.dataset_work = dataset_work
        self.roboflow_config = (roboflow_workspace, roboflow_project, roboflow_version)
        self.force_download = force_download
        self.split_ratios = self._validate_split_ratios(split_ratios)
        self.class_config = class_config
        self.target_formats = target_formats
        self.seed = seed
        self.run_id = run_id
        self.save_dir = Path(save_dir)
        self.max_samples = max_samples
        self.logger = self._setup_logger()

        # Инициализируем аугментации
        self.aug_pipeline = self._initialize_augmentations(augmentation_config)

        os.makedirs(self.dataset_src, exist_ok=True)
        os.makedirs(self.dataset_work, exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Настройка системы логирования"""
        logger_name = f"{self.__class__.__name__}_{self.run_id or 'main'}"

        # Проверяем, существует ли уже логгер с таким именем
        if logger_name in logging.root.manager.loggerDict:
            return logging.getLogger(logger_name)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        # Очищаем существующие обработчики
        logger.handlers = []

        # Создаем handler для вывода в консоль
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Создаем директорию для логов если её нет
        log_dir = self.save_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем handler для записи в файл
        file_handler = logging.FileHandler(
            log_dir /
            f'run_{self.run_id or "main"}_{datetime.now():%Y%m%d_%H%M%S}.log'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Отключаем передачу логов родительским логгерам
        logger.propagate = False

        return logger

    def _validate_split_ratios(self, ratios: Tuple) -> Tuple:
        """Проверка корректности пропорций разделения"""
        if abs(sum(ratios) - 1.0) > 1e-3:
            raise ValueError("Сумма пропорций должна быть равна 1")
        return ratios

    def _initialize_augmentations(self, config: Optional[Dict] = None) -> A.Compose:
        """Инициализация пайплайна аугментаций"""
        if config is None:
            # Дефолтные аугментации
            transforms = [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.4),
                A.Transpose(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.RandomGamma(p=0.2),
                A.Blur(p=0.1),
                A.CLAHE(p=0.2),
                A.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.5)
            ]
        else:
            # Пользовательские аугментации из конфига
            transforms = [getattr(A, name)(**params)
                          for name, params in config.items()]

        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='coco',
                min_visibility=0.3,
                label_fields=['class_labels']
            )
        )

    def _download_dataset(self) -> None:
        """Загрузка датасета из Roboflow с корректной обработкой структуры"""
        try:
            if not all(self.roboflow_config):
                raise ValueError(
                    "Требуется указать workspace, project и version для Roboflow")

            rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
            project = rf.workspace(self.roboflow_config[0]).project(
                self.roboflow_config[1])

            # Создаем временную папку для загрузки
            temp_dir = os.path.join(self.dataset_src, "temp_download")
            os.makedirs(temp_dir, exist_ok=True)

            # Скачивание датасета
            dataset = project.version(self.roboflow_config[2]).download(
                "coco",
                location=temp_dir,
                overwrite=self.force_download
            )

            # Создаем целевую версионную папку
            version_folder = os.path.join(
                self.dataset_src,
                f"{self.roboflow_config[1]}-{self.roboflow_config[2]}"
            )
            if os.path.exists(version_folder):
                shutil.rmtree(version_folder)
            os.makedirs(version_folder, exist_ok=True)

            # Поиск фактической папки с данными (Roboflow может создавать вложенные папки)
            data_root = temp_dir
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    data_root = item_path
                    break

            # Перенос всех файлов и подпапок в версионную папку
            for item in os.listdir(data_root):
                src = os.path.join(data_root, item)
                dst = os.path.join(version_folder, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            self.logger.info(f"Датасет успешно сохранен в: {version_folder}")

        except Exception as e:
            self.logger.error(f"Ошибка загрузки датасета: {str(e)}")
            raise
        finally:
            # Очистка временной папки
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def _copy_dataset_to_workdir(self) -> None:
        """Перенос данных в рабочую директорию с правильной структурой"""
        try:
            version_folder = os.path.join(
                self.dataset_src,
                f"{self.roboflow_config[1]}-{self.roboflow_config[2]}"
            )

            # Создаем базовую структуру рабочей директории
            work_data_dir = os.path.join(self.dataset_work, "data")
            os.makedirs(work_data_dir, exist_ok=True)

            # Копирование всех изображений напрямую в work/data
            image_extensions = ('.png', '.jpg', '.jpeg')
            for root, _, files in os.walk(version_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        src_path = os.path.join(root, file)
                        dst_path = os.path.join(work_data_dir, file)
                        if not os.path.exists(dst_path):
                            shutil.copy(src_path, dst_path)

            # Обработка аннотаций
            annotation_files = [f for f in os.listdir(
                version_folder) if f.endswith(".json")]
            if not annotation_files:
                raise FileNotFoundError("Не найден файл аннотаций COCO")

            src_ann_path = os.path.join(version_folder, annotation_files[0])
            dst_ann_path = os.path.join(self.dataset_work, "annotations.json")

            with open(src_ann_path, 'r') as f_src, open(dst_ann_path, 'w') as f_dst:
                coco_data = json.load(f_src)

                # Расширенная проверка структуры данных
                if not coco_data.get("annotations"):
                    raise ValueError(
                        "Загруженные аннотации не содержат данных")
                if not coco_data.get("categories"):
                    raise ValueError("Загруженные категории отсутствуют")

                # Корректировка путей изображений
                for img_info in coco_data["images"]:
                    img_info["file_name"] = os.path.basename(
                        img_info["file_name"])

                json.dump(coco_data, f_dst, indent=2)

            self._create_yolo_yaml()
            self.logger.info("Рабочая директория успешно подготовлена")

        except Exception as e:
            self.logger.error(
                f"Ошибка подготовки рабочей директории: {str(e)}")
            raise

    def _filter_and_merge_classes(self) -> None:
        """Фильтрация и объединение классов с улучшенной диагностикой"""
        if not self.class_config:
            self.logger.info(
                "Конфигурация классов не задана, пропускаем фильтрацию классов")
            return

        self.logger.info("Начало обработки классов...")
        try:
            # Загрузка аннотаций для предварительной проверки
            with open(os.path.join(self.dataset_work, "annotations.json"), 'r') as f:
                raw_coco = json.load(f)

            # Проверка наличия аннотаций до загрузки в FiftyOne
            if not raw_coco.get("annotations"):
                raise ValueError(
                    "Исходные аннотации не содержат объектов. "
                    "Проверьте: \n1. Корректность экспорта из Roboflow\n"
                    "2. Наличие разметки в датасете\n3. Соответствие версии датасета"
                )

                # Проверка структуры директорий и файлов перед загрузкой
            if not os.path.exists(self.dataset_work):
                raise ValueError(
                    f"Директория {self.dataset_work} не существует")

            data_dir = os.path.join(self.dataset_work, "data")
            if not os.path.exists(data_dir):
                raise ValueError(
                    f"Директория с изображениями {data_dir} не существует")

            ann_path = os.path.join(self.dataset_work, "annotations.json")
            if not os.path.exists(ann_path):
                raise ValueError(f"Файл аннотаций {ann_path} не найден")

            # Проверка содержимого директории с изображениями
            image_files = [f for f in os.listdir(data_dir)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not image_files:
                raise ValueError(f"В директории {data_dir} нет изображений")

            # Загрузка и проверка аннотаций
            with open(ann_path, 'r') as f:
                annotations = json.load(f)
                if not annotations.get('images') or not annotations.get('annotations'):
                    raise ValueError(
                        "Файл аннотаций пуст или имеет неверный формат")

            # Загрузка датасета
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.COCODetectionDataset,
                data_path=data_dir,
                labels_path=ann_path,
                include_id=True,
            )

            # Вывод всех полей датасета для диагностики
            self.logger.info("Доступные поля датасета:")
            if len(dataset) > 0:
                first_sample = dataset.first()
                self.logger.info(f"Все поля: {first_sample.field_names}")
                self.logger.info(
                    f"Поля с данными: {[f for f in first_sample.field_names if getattr(first_sample, f) is not None]}")
            else:
                self.logger.warning("Датасет пуст, невозможно показать поля")

            # Проверка загрузки
            if len(dataset) == 0:
                raise ValueError("Датасет пуст после загрузки")

            # Сбор классов с сохранением оригинальных имен
            all_classes = set()
            for sample in dataset:
                if sample.detections is not None:
                    for det in sample.detections.detections:
                        all_classes.add(det.label)

            self.logger.info(f"Найденные классы: {all_classes}")

            # Валидация class_config без модификации имен
            class_mapping = {
                src: target
                for target, sources in self.class_config.items()
                for src in sources
            }

            missing_classes = [
                c for c in class_mapping.keys() if c not in all_classes]
            if missing_classes:
                raise ValueError(
                    f"Классы из конфига отсутствуют в аннотациях: {missing_classes}\n"
                    f"Доступные классы: {list(all_classes)}"
                )

            # Создаем словарь для маппинга имен классов на их ID
            class_to_id = {class_mapping[src]: i + 1
                           for i, src in enumerate(set(class_mapping.keys()))}

            # Применение маппинга классов
            filtered_annotations = []
            for sample in tqdm(dataset.iter_samples(autosave=True), desc="Обработка классов"):
                if sample.detections is not None:
                    new_detections = []
                    for detection in sample.detections.detections:
                        if detection.label in class_mapping:
                            new_label = class_mapping[detection.label]
                            detection.label = class_mapping[detection.label]
                            new_detections.append(detection)

                            # Создаем новую COCO аннотацию
                            bbox = detection.bounding_box
                            filtered_annotations.append({
                                "id": len(filtered_annotations) + 1,
                                "image_id": sample.coco_id,
                                # Используем предварительно созданный маппинг
                                "category_id": class_to_id[new_label],
                                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0
                            })

                    sample.detections.detections = new_detections
                    sample.save()

            # Обновляем COCO аннотации
            coco_path = os.path.join(self.dataset_work, "annotations.json")
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            # Обновляем категории и аннотации
            unique_categories = list(set(class_mapping.values()))
            coco_data["categories"] = [
                {"id": class_to_id[cat_name], "name": cat_name}
                for cat_name in unique_categories
            ]
            coco_data["annotations"] = filtered_annotations

            # Сохраняем обновленные аннотации
            with open(coco_path, 'w') as f:
                json.dump(coco_data, f, indent=2)

            self.logger.info(
                f"Фильтрация классов завершена. "
                f"Сохранено {len(filtered_annotations)} аннотаций "
                f"для {len(unique_categories)} классов"
            )

        except Exception as e:
            self.logger.error(f"Критическая ошибка фильтрации: {str(e)}")
            raise

    def _update_coco_categories(self, dataset: fo.Dataset) -> None:
        """Обновление категорий в COCO аннотациях"""
        categories = list(OrderedDict.fromkeys(
            [det.label for sample in dataset if sample.detections is not None for det in sample.detections.detections]
        ))

        coco_path = os.path.join(self.dataset_work, "annotations.json")
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        new_categories = [{"id": i+1, "name": name}
                          for i, name in enumerate(categories)]
        id_mapping = {old["id"]: i+1 for old in coco_data["categories"]
                      for i, name in enumerate(categories) if old["name"] == name}

        for ann in tqdm(coco_data["annotations"], desc="Обновление аннотаций"):
            ann["category_id"] = id_mapping.get(ann["category_id"], 0)

        coco_data["categories"] = new_categories
        coco_data["annotations"] = [
            ann for ann in coco_data["annotations"] if ann["category_id"] != 0]

        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)
        self.logger.info("COCO аннотации обновлены")

    def _split_dataset(self) -> None:
        """Стратифицированное разделение данных с проверкой целостности файлов"""
        try:
            self.logger.info("Начало разделения датасета...")

            # Загрузка аннотаций COCO
            coco_path = os.path.join(self.dataset_work, "annotations.json")
            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            # Применяем ограничение на количество изображений если задано max_samples
            if self.max_samples is not None:
                np.random.seed(self.seed)
                total_images = len(coco_data["images"])
                if total_images > self.max_samples:
                    self.logger.info(f"Ограничение датасета до {self.max_samples} изображений из {total_images}")
                    
                    # Случайно выбираем изображения
                    selected_images = np.random.choice(
                        coco_data["images"], 
                        size=self.max_samples, 
                        replace=False
                    ).tolist()
                    
                    # Получаем ID выбранных изображений
                    selected_image_ids = {img["id"] for img in selected_images}
                    
                    # Фильтруем аннотации
                    coco_data["annotations"] = [
                        ann for ann in coco_data["annotations"]
                        if ann["image_id"] in selected_image_ids
                    ]
                    coco_data["images"] = selected_images

            # Усиленная проверка данных
            required_fields = ["images", "annotations", "categories"]
            for field in required_fields:
                if field not in coco_data:
                    raise ValueError(f"Отсутствует обязательное поле {field}")

            if not coco_data["images"]:
                raise ValueError("Список изображений пуст")
            if not coco_data["annotations"]:
                raise ValueError("Список аннотаций пуст")
            if not coco_data["categories"]:
                raise ValueError("Список категорий пуст")

            # Сбор информации для стратификации
            image_ids = [img["id"] for img in coco_data["images"]]
            annotations = defaultdict(list)
            for ann in coco_data["annotations"]:
                annotations[ann["image_id"]].append(ann["category_id"])

            # Создание страт
            strata = []
            for img_id in image_ids:
                cats = annotations.get(img_id, [])
                strata.append(max(set(cats), key=cats.count)
                              if cats else "empty")

            # Добавляем анализ количества элементов в каждом классе
            class_counts = {}
            for stratum in strata:
                if stratum not in class_counts:
                    class_counts[stratum] = 0
                class_counts[stratum] += 1

            self.logger.info("Количество элементов в каждом классе:")
            for class_id, count in class_counts.items():
                if class_id == "empty":
                    self.logger.info(f"Изображения без аннотаций: {count}")
                else:
                    category_name = next(
                        (cat["name"] for cat in coco_data["categories"] if cat["id"] == class_id), f"Класс {class_id}")
                    self.logger.info(f"{category_name}: {count} изображений")

            # Разделение на три части за один раз
            n_samples = len(image_ids)
            train_size = self.split_ratios[0]
            val_size = self.split_ratios[1]

            # Используем numpy для разделения с сохранением стратификации
            unique_strata = np.unique(strata)
            train_idx = []
            val_idx = []
            test_idx = []

            # Устанавливаем seed для воспроизводимости
            np.random.seed(self.seed)

            for stratum in unique_strata:
                if stratum == "empty":
                    continue

                idx = np.where(np.array(strata) == stratum)[0]
                np.random.shuffle(idx)

                n_stratum = len(idx)
                n_train = int(n_stratum * train_size)
                n_val = int(n_stratum * val_size)

                train_idx.extend(idx[:n_train])
                val_idx.extend(idx[n_train:n_train + n_val])
                test_idx.extend(idx[n_train + n_val:])

            # Перемешиваем индексы
            np.random.shuffle(train_idx)
            np.random.shuffle(val_idx)
            np.random.shuffle(test_idx)

            # Формирование итоговых сплитов
            train_ids = np.array(image_ids)[train_idx]
            val_ids = np.array(image_ids)[val_idx]
            test_ids = np.array(image_ids)[test_idx]

            # Создание директорий для сплитов
            splits = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids
            }

            for split_name, split_ids in splits.items():
                split_dir = os.path.join(self.dataset_work, split_name)
                os.makedirs(os.path.join(split_dir, "data"), exist_ok=True)

                # Копирование изображений
                for img_id in split_ids:
                    img_info = next(
                        img for img in coco_data["images"] if img["id"] == img_id)
                    src_path = os.path.join(
                        self.dataset_work, "data", img_info["file_name"])
                    dst_path = os.path.join(
                        split_dir, "data", img_info["file_name"])
                    shutil.copy2(src_path, dst_path)

                # Создание COCO аннотаций для сплита
                split_annotations = {
                    "images": [img for img in coco_data["images"] if img["id"] in split_ids],
                    "annotations": [ann for ann in coco_data["annotations"] if ann["image_id"] in split_ids],
                    "categories": coco_data["categories"]
                }

                with open(os.path.join(split_dir, "annotations.json"), 'w') as f:
                    json.dump(split_annotations, f, indent=2)

            # Добавляем информацию о распределении классов в каждом сплите
            for split_name, split_ids in splits.items():
                split_strata = [strata[image_ids.index(
                    img_id)] for img_id in split_ids]
                split_counts = {}
                for stratum in split_strata:
                    if stratum not in split_counts:
                        split_counts[stratum] = 0
                    split_counts[stratum] += 1

                self.logger.info(f"\nРаспределение классов в {split_name}:")
                for class_id, count in split_counts.items():
                    if class_id == "empty":
                        self.logger.info(f"Изображения без аннотаций: {count}")
                    else:
                        category_name = next(
                            (cat["name"] for cat in coco_data["categories"] if cat["id"] == class_id), f"Класс {class_id}")
                        self.logger.info(
                            f"{category_name}: {count} изображений")

            self.logger.info(
                f"\nИтоговое разделение датасета:\n"
                f"Train: {len(train_ids)} изображений\n"
                f"Val: {len(val_ids)} изображений\n"
                f"Test: {len(test_ids)} изображений"
            )

        except Exception as e:
            self.logger.error(f"Критическая ошибка разделения: {str(e)}")
            raise

    def _reorganize_split_dir(self, split_dir: str) -> None:
        """Исправление структуры директорий после экспорта"""
        try:
            # Переименование папок
            os.rename(
                os.path.join(split_dir, "images"),
                os.path.join(split_dir, "data")
            )

            # Обновление путей в аннотациях
            coco_path = os.path.join(split_dir, "annotations.json")
            with open(coco_path, 'r') as f:
                data = json.load(f)

            for img in data["images"]:
                img["file_name"] = os.path.join(
                    "data", os.path.basename(img["file_name"]))

            with open(coco_path, 'w') as f:
                json.dump(data, f)

        except Exception as e:
            self.logger.warning(f"Ошибка реорганизации {split_dir}: {str(e)}")

    def _apply_augmentations(self) -> None:
        """Применение аугментаций с потоковой обработкой"""
        try:
            self.logger.info("Применение аугментаций...")
            cv2.destroyAllWindows()
            cv2.setNumThreads(2)

            train_dir = os.path.join(self.dataset_work, "train")
            coco = COCO(os.path.join(train_dir, "annotations.json"))

            # Проверка наличия аннотаций и изображений
            if not coco.dataset["annotations"]:
                self.logger.warning(
                    "Нет аннотаций для аугментации. Пропуск этапа.")
                return

            if not coco.dataset["images"]:
                self.logger.warning(
                    "Нет изображений для аугментации. Пропуск этапа.")
                return

            # Инициализация ID с проверкой на пустые коллекции
            image_ids = [img["id"] for img in coco.dataset["images"]]
            ann_ids = [ann["id"] for ann in coco.dataset["annotations"]]

            image_id_offset = max(image_ids) + 1 if image_ids else 0
            ann_id_offset = max(ann_ids) + 1 if ann_ids else 0

            new_images = []
            new_annotations = []

            for img_info in tqdm(coco.dataset["images"], desc="Аугментация"):
                img_path = os.path.join(
                    train_dir, "data", img_info["file_name"])
                if not os.path.exists(img_path):
                    self.logger.warning(
                        f"Изображение {img_path} не найдено, пропускаем")
                    continue

                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                anns = coco.loadAnns(coco.getAnnIds(imgIds=img_info["id"]))

                # Пропуск изображений без аннотаций
                if not anns:
                    continue

                # Подготовка данных для аугментации
                bboxes = []
                category_ids = []
                for ann in anns:
                    x, y, w, h = ann["bbox"]
                    bboxes.append([x, y, w, h])
                    category_ids.append(ann["category_id"])

                # Применение аугментации с правильными параметрами
                try:
                    transformed = self.aug_pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=category_ids  # Только необходимые параметры
                    )

                    # Проверяем, что после аугментации остались боксы
                    if transformed["bboxes"]:
                        new_img_info = {
                            "id": image_id_offset,
                            "file_name": f"aug_{image_id_offset}_{img_info['file_name']}",
                            "width": transformed["image"].shape[1],
                            "height": transformed["image"].shape[0],
                            **{k: v for k, v in img_info.items() if k not in ("id", "file_name", "width", "height")}
                        }

                        cv2.imwrite(
                            os.path.join(train_dir, "data",
                                         new_img_info["file_name"]),
                            cv2.cvtColor(
                                transformed["image"], cv2.COLOR_RGB2BGR)
                        )

                        # Создание новых аннотаций
                        for bbox, category_id in zip(transformed["bboxes"], transformed["class_labels"]):
                            new_ann = {
                                "id": ann_id_offset,
                                "image_id": image_id_offset,
                                "category_id": category_id,
                                "bbox": [round(float(x), 2) for x in bbox],
                                "area": bbox[2] * bbox[3],
                                "iscrowd": 0
                            }
                            new_annotations.append(new_ann)
                            ann_id_offset += 1

                        new_images.append(new_img_info)
                        image_id_offset += 1

                except Exception as aug_error:
                    self.logger.warning(
                        f"Ошибка аугментации изображения {img_info['file_name']}: {str(aug_error)}")
                    continue

            # Обновление аннотаций только при наличии новых данных
            if new_images and new_annotations:
                coco.dataset["images"].extend(new_images)
                coco.dataset["annotations"].extend(new_annotations)
                with open(os.path.join(train_dir, "annotations.json"), 'w') as f:
                    json.dump(coco.dataset, f)
                self.logger.info(
                    f"Сгенерировано {len(new_images)} аугментированных изображений")
            else:
                self.logger.info("Нет новых данных для сохранения")

        except Exception as e:
            self.logger.error(f"Ошибка аугментации: {str(e)}")
            raise

    def _convert_annotations(self) -> None:
        """Конвертация аннотаций в целевые форматы"""
        self.logger.info("Начало конвертации аннотаций...")
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(self.dataset_work, split)

            if "yolo" in self.target_formats:
                self._convert_to_yolo(split_dir)

            if "coco" in self.target_formats:
                self._preserve_coco(split_dir)
        self.logger.info("Конвертация завершена")

    def _convert_to_yolo(self, split_dir: str) -> None:
        """Конвертация в формат YOLO"""
        self.logger.info(f"Конвертация {split_dir} в YOLO формат...")
        coco = COCO(os.path.join(split_dir, "annotations.json"))
        os.makedirs(os.path.join(split_dir, "labels"), exist_ok=True)

        category_map = {cat["id"]: i for i,
                        cat in enumerate(coco.dataset["categories"])}

        for img_info in tqdm(coco.dataset["images"], desc="Конвертация в YOLO"):
            ann_ids = coco.getAnnIds(imgIds=img_info["id"])
            anns = coco.loadAnns(ann_ids)

            yolo_lines = []
            for ann in anns:
                # COCO: [x_min, y_min, width, height] - уже нормализованные значения
                x_min, y_min, width, height = ann["bbox"]

                # Проверка корректности входных данных
                if width <= 0 or height <= 0:
                    self.logger.warning(
                        f"Пропуск некорректного бокса в изображении {img_info['file_name']}: width={width}, height={height}")
                    continue

                # YOLO: [x_center, y_center, width, height]
                # Просто преобразуем координаты из (x_min, y_min) в (x_center, y_center)
                x_center = x_min + width/2
                y_center = y_min + height/2

                # Проверка нормализованных значений
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                    self.logger.warning(
                        f"Подозрительные нормализованные значения в {img_info['file_name']}: "
                        f"x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}"
                    )

                yolo_lines.append(
                    f"{category_map[ann['category_id']]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

            # Добавим логирование для отладки
            if not yolo_lines:
                self.logger.warning(
                    f"Нет аннотаций для изображения {img_info['file_name']}")
                continue

            txt_path = os.path.join(split_dir, "labels", os.path.splitext(
                img_info["file_name"])[0] + ".txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))

        self._create_yolo_yaml()

    def _create_yolo_yaml(self) -> None:
        """Генерация YAML-конфига для YOLO с актуальными путями"""
        yaml_path = os.path.join(self.dataset_work, "dataset.yaml")

        with open(os.path.join(self.dataset_work, "annotations.json"), 'r') as f:
            coco_data = json.load(f)

        # Используем оригинальные имена категорий
        class_names = [cat["name"] for cat in sorted(
            coco_data["categories"], key=lambda x: x["id"])]

        config = {
            "path": os.path.abspath(self.dataset_work),
            "train": "train/data",
            "val": "val/data",
            "test": "test/data",
            "names": {i: name for i, name in enumerate(class_names)},
            "nc": len(class_names),
            # Добавляем правильные относительные пути
            "train_images": "train/data",
            "val_images": "val/data",
            "test_images": "test/data"
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        self.logger.info(f"Создан YAML-конфиг для YOLO: {yaml_path}")

    def _preserve_coco(self, split_dir: str) -> None:
        """Сохранение COCO формата"""
        coco_path = os.path.join(split_dir, "annotations.json")
        with open(coco_path, 'r') as f:
            data = json.load(f)

        for img in data["images"]:
            img["file_name"] = os.path.join("images", img["file_name"])

        with open(coco_path, 'w') as f:
            json.dump(data, f)
        self.logger.info(f"COCO аннотации для {split_dir} обновлены")

    def _validate_dataset(self) -> None:
        """Расширенная валидация с логированием структуры"""
        self.logger.info("Запуск расширенной валидации...")

        # Проверка структуры аннотаций
        coco_path = os.path.join(self.dataset_work, "annotations.json")
        with open(coco_path, 'r') as f:
            coco_data = json.load(f)

        self.logger.info(
            "Статистика аннотаций:\n"
            f"- Изображений: {len(coco_data['images'])}\n"
            f"- Аннотаций: {len(coco_data['annotations'])}\n"
            f"- Категорий: {len(coco_data['categories'])}\n"
            f"- Примеры категорий: {[c['name'] for c in coco_data['categories'][:3]]}"
        )

        """Проверка целостности данных"""
        self.logger.info("Валидация датасета...")
        for split in ["train", "val", "test"]:
            try:
                split_dir = os.path.join(self.dataset_work, split)

                # Проверка существования основных файлов
                required_files = {
                    "annotations": os.path.join(split_dir, "annotations.json"),
                    "data_dir": os.path.join(split_dir, "data")
                }

                for name, path in required_files.items():
                    if not os.path.exists(path):
                        raise FileNotFoundError(
                            f"Не найден требуемый файл/директория: {path}")

                # Базовые проверки через FiftyOne
                dataset = fo.Dataset.from_dir(
                    dataset_dir=split_dir,
                    dataset_type=fo.types.COCODetectionDataset
                )

                # Проверка метаданных
                dataset.compute_metadata()

                # Кастомные проверки
                self._perform_custom_validation(dataset, split)

                self.logger.info(
                    f"{split.capitalize()} набор прошел базовые проверки")

            except Exception as e:
                self.logger.error(f"Ошибка валидации {split}: {str(e)}")
                raise

    def _perform_custom_validation(self, dataset: fo.Dataset, split: str) -> None:
        """Дополнительные проверки целостности данных с физическим подсчетом файлов"""
        split_dir = os.path.join(self.dataset_work, split)
        data_dir = os.path.join(split_dir, "data")

        try:
            # 1. Проверка физического наличия файлов
            image_files = [f for f in os.listdir(data_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            actual_files_count = len(image_files)

            # 2. Проверка аннотаций COCO
            ann_file = os.path.join(split_dir, "annotations.json")
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)

                # Исправляем пути в аннотациях перед проверкой
                for img in coco_data["images"]:
                    img["file_name"] = os.path.basename(img["file_name"])

                # Сохраняем исправленные аннотации
                with open(ann_file, 'w') as f_out:
                    json.dump(coco_data, f_out, indent=2)

                expected_files_count = len(coco_data["images"])

            # 3. Загружаем датасет заново после исправления путей
            dataset = fo.Dataset.from_dir(
                dataset_type=fo.types.COCODetectionDataset,  # Изменяем порядок и параметры
                data_path=data_dir,
                labels_path=ann_file,
                include_id=True
            )

            # 4. Сравнение количеств
            if actual_files_count != expected_files_count:
                raise ValueError(
                    f"Несоответствие в {split}: "
                    f"файлов в папке {actual_files_count}, "
                    f"в аннотациях {expected_files_count}"
                )

            # 5. Проверка соответствия имен файлов
            annotated_files = {img["file_name"] for img in coco_data["images"]}
            missing_in_ann = set(image_files) - annotated_files
            missing_in_fs = annotated_files - set(image_files)

            if missing_in_ann:
                self.logger.warning(
                    f"В {split} обнаружены файлы без аннотаций: {len(missing_in_ann)}")

            if missing_in_fs:
                raise FileNotFoundError(
                    f"В {split} отсутствуют файлы из аннотаций: {list(missing_in_fs)[:5]}..."
                )

            # 6. Проверка FiftyOne Dataset
            if len(dataset) != expected_files_count:
                raise ValueError(
                    f"FiftyOne загрузил {len(dataset)} изображений, "
                    f"ожидалось {expected_files_count}"
                )

            self.logger.info(
                f"{split} валидация пройдена: {actual_files_count} изображений")

        except Exception as e:
            self.logger.error(f"Ошибка валидации {split}: {str(e)}")
            raise

    def get_class_names(self) -> List[str]:
        """Получение списка классов"""
        with open(os.path.join(self.dataset_work, "train", "annotations.json"), 'r') as f:
            return [cat["name"] for cat in json.load(f)["categories"]]

    def get_dataset_stats(self) -> Dict:
        """Статистика датасета"""
        stats = {}
        for split in ["train", "val", "test"]:
            coco_path = os.path.join(
                self.dataset_work, split, "annotations.json")
            with open(coco_path, 'r') as f:
                data = json.load(f)
                stats[split] = {
                    "images": len(data["images"]),
                    "annotations": len(data["annotations"]),
                    "categories": len(data["categories"])
                }
        return stats

    def debug_coco_to_yolo_conversion(self, split_dir: str) -> None:
        """Отладочная функция для проверки конвертации координат"""
        coco = COCO(os.path.join(split_dir, "annotations.json"))

        # Возьмем первые несколько аннотаций для проверки
        for img_info in list(coco.dataset["images"])[:5]:
            print(f"\nПроверка изображения: {img_info['file_name']}")
            print(
                f"Размеры изображения: {img_info['width']}x{img_info['height']}")

            ann_ids = coco.getAnnIds(imgIds=img_info["id"])
            anns = coco.loadAnns(ann_ids)

            for ann in anns:
                x_min, y_min, width, height = ann["bbox"]
                print("\nCOCO координаты:")
                print(
                    f"x_min: {x_min}, y_min: {y_min}, width: {width}, height: {height}")

                # Проверяем, не нормализованы ли уже координаты
                if all(0 <= coord <= 1 for coord in [x_min, y_min, width, height]):
                    print("ВНИМАНИЕ: Похоже координаты уже нормализованы!")

                # Вычисляем YOLO координаты
                x_center = (x_min + width/2) / float(img_info["width"])
                y_center = (y_min + height/2) / float(img_info["height"])
                norm_width = width / float(img_info["width"])
                norm_height = height / float(img_info["height"])

                print("\nYOLO координаты:")
                print(f"x_center: {x_center:.6f}")
                print(f"y_center: {y_center:.6f}")
                print(f"width: {norm_width:.6f}")
                print(f"height: {norm_height:.6f}")

    def visualize_sample(self, split: str = "train") -> None:
        """Визуализация данных через FiftyOne"""
        self.logger.info(f"Визуализация набора {split}...")
        dataset = fo.Dataset.from_dir(
            dataset_dir=os.path.join(self.dataset_work, split),
            dataset_type=fo.types.COCODetectionDataset
        )
        session = fo.launch_app(dataset)
        session.wait()

    def prepare_dataset(self) -> None:
        """Основной пайплайн обработки с улучшенной обработкой ошибок"""
        self.logger.info("Начало подготовки датасета...")
        try:
            if not os.listdir(self.dataset_src) or self.force_download:
                self._download_dataset()

            self._copy_dataset_to_workdir()
            self._filter_and_merge_classes()
            self._split_dataset()
            # self._apply_augmentations()
            self._convert_annotations()
            self._validate_dataset()

            stats = self.get_dataset_stats()
            self.logger.info("\nПодготовка данных завершена успешно!")
            self.logger.info(f"Структура датасета:\n"
                             f"Train: {stats['train']['images']} изображений, {stats['train']['annotations']} аннотаций\n"
                             f"Val: {stats['val']['images']} изображений, {stats['val']['annotations']} аннотаций\n"
                             f"Test: {stats['test']['images']} изображений, {stats['test']['annotations']} аннотаций\n"
                             f"Классы: {self.get_class_names()}")
        except Exception as e:
            self.logger.error(f"Критическая ошибка: {str(e)}", exc_info=True)
            raise RuntimeError(
                f"Не удалось подготовить датасет: {str(e)}") from e


preprocessor = HelminthDatasetPreprocessor(
    #force_download=True,
    roboflow_workspace="ai-sy0nz",
    roboflow_project="combined-helminth-eggs-dataset",
    roboflow_version=21,
    # class_config={"egg": ["trematode-helminth-egg", "helminth-egg"]},
    class_config={
        "trematode-helminth-egg": ["trematode-helminth-egg"], "helminth-egg": ["helminth-egg"]},
    augmentation_config={
        "RandomRotate90": {"p": 0.5},
        "ColorJitter": {"p": 0.5}
    },
    target_formats=["yolo", "coco"],
    seed=42,
    run_id=1,  # Добавляем run_id
    save_dir="dataset_log",  # Добавляем save_dir
    max_samples=150000 # Добавляем max_samples
)

# Перемещение папок
def move_directories():
    shutil.move("dataset_work/train/data", "dataset_work/train/images")
    shutil.move("dataset_work/val/data", "dataset_work/val/images")
    shutil.move("dataset_work/test/data", "dataset_work/test/images")

# Редактирование YAML-файла
def edit_yaml_file(file_path):
    # Чтение содержимого файла
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Флаги для удаления строк
    remove_train_images = False
    remove_val_images = False
    remove_test_images = False

    # Обработка каждой строки
    updated_lines = []
    for line in lines:
        if 'train_images: train/data' in line:
            remove_train_images = True
        elif 'val_images: val/data' in line:
            remove_val_images = True
        elif 'test_images: test/data' in line:
            remove_test_images = True
        else:
            # Замена путей в строках
            line = line.replace('train: train/data', 'train: train/images')
            line = line.replace('val: val/data', 'val: val/images')
            line = line.replace('test: test/data', 'test: test/images')
            updated_lines.append(line)

    # Если строки не были удалены, добавляем их в список для удаления
    if remove_train_images or remove_val_images or remove_test_images:
        with open(file_path, 'w') as file:
            file.writelines(updated_lines)


# Основной блок выполнения
if __name__ == "__main__":
    try:
        #основная функция скачивания/подготовки датасета
        preprocessor.prepare_dataset()
        # Вызов функции перемещения папок
        print ("Реогранизация папок")
        move_directories()
        
        # Редактирование YAML-файла
        yaml_file_path = "dataset_work/dataset.yaml"
        if os.path.exists(yaml_file_path):
            edit_yaml_file(yaml_file_path)
        else:
            print(f"Файл {yaml_file_path} не найден.")
    
    except Exception as e:
        print(f"Произошла ошибка: {e}")
    print ("Датасет подготовлен к работе")
