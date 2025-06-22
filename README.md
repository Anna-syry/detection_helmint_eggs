python3 -m pip install -r requirements.txt
python3 -m pip install torchmetrics
python3 -m pip install aiofiles==22.1.0 packaging==23 optuna wandb python-dotenv
python3 -m pip install -U tensorboard tensorboard-plugin-profile torch-tb-profiler


#scikit-optimize

python3 datasets.py
python3 convert_annotations.py

==============================================================================
YOLO
==============================================================================
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt
python3 yolo_train.py


==============================================================================
Faster-CNN + ResNet
==============================================================================
apt install -y git
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
#git clone https://github.com/facebookresearch/detectron2.git

cd rcnn_resnet

#python3 modified_training.py --model faster_rcnn_r50
#python3 modified_training.py --model faster_rcnn_r101
python3 resnet_train.py --model faster_rcnn_x101

#tensorboard --logdir=/home/output --port=6006 --bind_all



==============================================================================
RT-DETRv3
==============================================================================
python RT-DETRv3/tools/train.py -c RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco-eggs.yml --eval -r /home/rtdetrv3_r50vd_6x.pdparams -o use_vdl=True vdl_log_dir=/home/output/vdl

visualdl --logdir /home/output/vdl --port 8040 --host 0.0.0.0

loss_class: 0.484807          # Потеря классификации
loss_bbox: 0.158414           # Потеря регрессии ограничивающих рамок
loss_giou: 0.189100           # Потеря GIoU
loss_class_aux: 3.889533      # Вспомогательная потеря классификации
loss_bbox_aux: 0.954726       # Вспомогательная потеря регрессии
loss_giou_aux: 1.156697       # Вспомогательная потеря GIoU
loss_class_dn: 0.293906       # Потеря классификации для denoising
loss_bbox_dn: 0.144063        # Потеря регрессии для denoising
loss_giou_dn: 0.173424        # Потеря GIoU для denoising
loss_class_aux_dn: 1.584356   # Вспомогательная потеря классификации для denoising
loss_bbox_aux_dn: 0.757643    # Вспомогательная потеря регрессии для denoising
loss_giou_aux_dn: 0.950892    # Вспомогательная потеря GIoU для denoising
loss: 20.526669               # Общая потеря (сумма всех компонентов)


 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.916
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.977
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.973
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.778
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.766
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.921
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.882
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.942
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.945
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.867
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.930
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.944
[03/21 23:00:16] ppdet.metrics.coco_utils INFO: Per-category of bbox AP: 
+--------------+-------+------------------------+-------+
| category     | AP    | category               | AP    |
+--------------+-------+------------------------+-------+
| helminth-egg | 0.935 | trematode-helminth-egg | 0.897 |
+--------------+-------+------------------------+-------+
[03/21 23:00:16] ppdet.metrics.coco_utils INFO: per-category PR curve has output to bbox_pr_curve folder.
[03/21 23:00:16] ppdet.engine.callbacks INFO: Total sample number: 1058, average FPS: 27.962482312408696






# Скрипт для запуска RT-DETRv3

```bash:run_rtdetrv3.sh
#!/bin/bash

# Скрипт для запуска обучения, оценки и инференса модели RT-DETRv3

# Установка переменных окружения
export CUDA_VISIBLE_DEVICES=0

# Создание директорий для выходных данных
mkdir -p /home/output/rtdetrv3_output
mkdir -p /home/output/rtdetrv3_eval
mkdir -p /home/output/rtdetrv3_infer

# Клонирование репозитория RT-DETRv3
if [ ! -d "/home/RT-DETRv3" ]; then
    git clone https://github.com/clxia12/RT-DETRv3.git /home/RT-DETRv3
    echo "Клонирован репозиторий RT-DETRv3"
else
    echo "Репозиторий RT-DETRv3 уже существует"
fi

# Установка зависимостей
pip install -r /home/RT-DETRv3/requirements.txt
pip install visualdl tqdm opencv-python

# Загрузка предобученных весов, если они еще не загружены
if [ ! -f "rtdetrv3_r50vd_6x.pdparams" ]; then
    echo "Загрузка предобученных весов..."
    wget https://paddledet.bj.bcebos.com/models/rtdetrv3_r50vd_6x.pdparams
    echo "Веса загружены"
else
    echo "Предобученные веса уже существуют"
fi

# Функция для обучения модели
train_model() {
    echo "Запуск обучения модели RT-DETRv3..."
    python train_rtdetrv3.py \
        --config /home/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
        --weights rtdetrv3_r50vd_6x.pdparams \
        --output_dir /home/output/rtdetrv3_output \
        --epochs 36 \
        --batch_size 2 \
        --learning_rate 0.0001 \
        --use_amp \
        --eval \
        --log_iter 50 \
        --save_interval 5 \
        --eval_interval 1
}

# Функция для оценки модели
evaluate_model() {
    echo "Запуск оценки модели RT-DETRv3..."
    python eval_rtdetrv3.py \
        --config /home/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
        --weights /home/output/rtdetrv3_output/model_final.pdparams \
        --output_dir /home/output/rtdetrv3_eval \
        --dataset val \
        --batch_size 2 \
        --use_vdl
}

# Функция для инференса модели
infer_model() {
    echo "Запуск инференса модели RT-DETRv3..."
    python infer_rtdetrv3.py \
        --config /home/RT-DETRv3/configs/rtdetrv3/rtdetrv3_r50vd_6x_coco.yml \
        --weights /home/output/rtdetrv3_output/model_final.pdparams \
        --output_dir /home/output/rtdetrv3_infer \
        --dataset test \
        --batch_size 1 \
        --threshold 0.5 \
        --save_images \
        --save_results
}

# Основная логика скрипта
case "$1" in
    train)
        train_model
        ;;
    eval)
        evaluate_model
        ;;
    infer)
        infer_model
        ;;
    all)
        train_model
        evaluate_model
        infer_model
        ;;
    *)
        echo "Использование: $0 {train|eval|infer|all}"
        echo "  train: только обучение модели"
        echo "  eval: только оценка модели"
        echo "  infer: только инференс модели"
        echo "  all: полный цикл (обучение, оценка, инференс)"
        exit 1
        ;;
esac

echo "Готово!"
```

## Инструкция по использованию

1. Сначала сделайте скрипт исполняемым:
   ```bash
   chmod +x run_rtdetrv3.sh
   ```

2. Запустите обучение модели:
   ```bash
   ./run_rtdetrv3.sh train
   ```

3. После обучения запустите оценку модели:
   ```bash
   ./run_rtdetrv3.sh eval
   ```

4. Для инференса на тестовом наборе данных:
   ```bash
   ./run_rtdetrv3.sh infer
   ```

5. Или запустите весь процесс сразу:
   ```bash
   ./run_rtdetrv3.sh all
   ```

## Примечания по запуску

1. Скрипт автоматически клонирует репозиторий RT-DETRv3, если он еще не существует.
2. Скрипт загружает предобученные веса, если они еще не загружены.
3. Для мониторинга обучения можно использовать VisualDL:
   ```bash
   visualdl --logdir=/home/output/rtdetrv3_output/vdl --port=8040 --host=0.0.0.0
   ```
4. Результаты инференса будут сохранены в директории `/home/output/rtdetrv3_infer/visualizations/`.
5. Метрики оценки будут сохранены в файле `/home/output/rtdetrv3_eval/eval_results.json`.

Все файлы (train_rtdetrv3.py, eval_rtdetrv3.py, infer_rtdetrv3.py и custom_metrics_logger.py) должны находиться в одной директории с run_rtdetrv3.sh.









































python3 train_optuna4.py --pin_memory --mixed_precision --cache_data --warmup 3 --save_roi 1 --optimizers AdamW --optuna_trials 10 --debug_mode

В директории debug_output находятся следующие поддиректории и файлы:
raw_images/
Содержит исходные изображения без боксов
Формат имен: epoch_XXX_batch_YYYYY_img_ZZZ_1_raw_iter_NNN.png
Где:
XXX: номер эпохи (3 цифры)
YYYYY: номер батча (5 цифр)
ZZZ: индекс изображения в батче (3 цифры)
NNN: номер итерации (3 цифры)
gt_boxes/
Содержит изображения с ground truth боксами (зеленого цвета)
Формат имен: epoch_XXX_batch_YYYYY_img_ZZZ_2_gt_iter_NNN.png
Боксы отображаются с метками классов
rpn_proposals/
Содержит изображения с предложениями от RPN (синего цвета)
Формат имен: epoch_XXX_batch_YYYYY_img_ZZZ_3_rpn_iter_NNN.png
Боксы отображаются со scores (уверенностью)
roi_features/
Содержит визуализации ROI features
Формат имен: epoch_XXX_batch_YYYYY_img_ZZZ_4_roi_features_iter_NNN.png
Отображает извлеченные признаки в виде сетки
final_detections/
Содержит изображения с финальными детекциями (красного цвета)
Формат имен: epoch_XXX_batch_YYYYY_img_ZZZ_5_final_iter_NNN.png
Боксы отображаются с метками классов и scores
debug_steps/
Содержит текстовые файлы с отладочной информацией
Формат имен: epoch_XXX_batch_YYYYY_debug.txt
Также содержит визуализации карт внимания: epoch_XXX_batch_YYYYY_attention.png
В каждой директории файлы теперь имеют уникальные имена благодаря добавлению суффикса _iter_NNN, что предотвращает их перезапись при повторных вызовах визуализации для одного и того же изображения.