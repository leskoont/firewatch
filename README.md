# firewatch

Детекция огня и дыма на видеопотоках с IP-камер и из видеофайлов. Модель YOLO11s,
обученная на открытом датасете D-Fire, на инференсе работает через onnxruntime без PyTorch.
При устойчивом срабатывании отправляет в Telegram кадр с разметкой и сохраняет снимок.

## Результаты

YOLO11s (предобучение COCO), 640x640, 30 эпох на RTX 2060 6 GB, лучшая эпоха - последняя
(mAP50 0.773 на val, [docs/results.csv](docs/results.csv)). Официальный тест D-Fire, `best.pt`
через `ultralytics val` ([docs/test_metrics_pt.json](docs/test_metrics_pt.json)):

| Класс | Изображений | Объектов | Precision | Recall | mAP50 | mAP50-95 |
|-------|-------------|----------|-----------|--------|-------|----------|
| all   | 4306 | 5189 | 0.759 | 0.687 | 0.760 | 0.436 |
| fire  | 1115 | 2878 | 0.718 | 0.616 | 0.703 | 0.367 |
| smoke | 2081 | 2311 | 0.800 | 0.758 | 0.817 | 0.506 |

### Тест без копий из train (ONNX)

Эти числа завышены: многие тестовые кадры из тех же роликов, что и train. По dHash (64 бита)
у 23.1% тестовых изображений есть кадр в train на расстоянии 0 бит, у 40.5% - не больше 2 бит.
`scripts/dedup_test.py` оставляет только изображения, далекие от всего train минимум на 7 бит
([docs/dedup_test.json](docs/dedup_test.json)). На полном тесте ONNX теряет к `best.pt` 0.002
mAP50 из-за квадратного входа ([docs/test_metrics_onnx.json](docs/test_metrics_onnx.json)).

| Тест | Изображений | mAP50 | mAP50-95 | fire mAP50 / 50-95 | smoke mAP50 / 50-95 |
|------|-------------|-------|----------|--------------------|---------------------|
| весь | 4306 | 0.758 | 0.432 | 0.699 / 0.365 | 0.816 / 0.499 |
| без копий из train | 1962 | 0.731 | 0.416 | 0.646 / 0.316 | 0.816 / 0.517 |

Один кадр при порогах по умолчанию (fire 0.40, smoke 0.50), без окна N из M:

| Тест | fire найден | smoke найден | ложная тревога на пустом кадре |
|------|-------------|--------------|--------------------------------|
| весь | 82.4% из 1115 | 72.8% из 2081 | 0.5% из 2005 |
| без копий из train | 79.5% из 711 | 77.5% из 1182 | 1.0% из 699 |

Огонь находится хуже дыма в основном из-за размера: на кадрах 640 px очаг часто занимает
несколько десятков пикселей, к тому же несколько очагов размечены то одним боксом, то разными.

![](docs/results.png)

### Скорость (ONNX)

`scripts/benchmark.py`: letterbox, инференс, NMS и перевод координат, 100 изображений из теста,
batch 1, FP32, onnxruntime 1.23.2, Windows 10 ([docs/bench.json](docs/bench.json)).

| Устройство | Провайдер | pre, мс | инференс, мс | post, мс | всего p50, мс | всего p95, мс | FPS |
|------------|-----------|---------|--------------|----------|---------------|---------------|-----|
| Intel Core i5-10400F | CPU | 3.2 | 90.4 | 0.3 | 94.0 | 99.6 | 10.6 |
| NVIDIA RTX 2060 6 GB | CUDA | 2.9 | 8.9 | 0.3 | 12.0 | 12.6 | 83.0 |

Смотреть каждый кадр не нужно: при 25 FPS и `process_every: 3` это ~8 инференсов в секунду,
i5-10400F тянет так одну камеру, с `process_every: 10` по расчету 3-4, дальше нужен GPU.

`scripts/check_onnx.py` сверяет экспорт с оригиналом на 50 тестовых изображениях: сырой выход
сети расходится не больше чем на 0.0066 px, `OnnxDetector` и ultralytics на том же .onnx
находят одни и те же 64 бокса. С .pt средний IoU 0.971: для .pt ultralytics дополняет кадр до
прямоугольника, кратного 32, а не до квадрата ([docs/check_onnx.json](docs/check_onnx.json)).

## Установка

Нужен Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate              # Linux: source .venv/bin/activate
pip install -e ".[dev]"             # рантайм и pytest
pip install -e ".[train]"           # ultralytics для обучения
```

## Данные и обучение

[D-Fire](https://github.com/gaiasd/DFireDataset): 21 527 изображений, классы `smoke` и `fire`,
разметка YOLO, CC0 1.0. `scripts/prepare_dfire.py` берет его из зеркала на Hugging Face
([badsaarow/d-fire](https://huggingface.co/datasets/badsaarow/d-fire), ~3.1 GB; оригинал на
OneDrive без браузера не скачать), уменьшает изображения до 640 px по длинной стороне и пишет
`data/dfire/dfire.yaml`. Официальный тест не трогается, val (10%) отделяется от train по хешу
имени файла: 15 508 / 1 713 / 4 306 изображений, в тесте 2 005 без огня и дыма. Боксы со
стороной меньше 0.1% кадра отбрасываются как ошибки разметки.

```bash
python scripts/prepare_dfire.py
python scripts/train.py                              # configs/train.yaml
python scripts/train.py epochs=10 batch=8            # переопределить любой параметр
python scripts/train.py --keep-cuda-cache --resume runs/yolo11s-dfire/weights/last.pt batch=8
python scripts/evaluate.py runs/yolo11s-dfire/weights/best.pt --split test --out docs/test_metrics_pt.json
python scripts/export_onnx.py runs/yolo11s-dfire/weights/best.pt   # -> models/firewatch-yolo11s.onnx
python scripts/check_onnx.py runs/yolo11s-dfire/weights/best.pt models/firewatch-yolo11s.onnx
python scripts/benchmark.py models/firewatch-yolo11s.onnx --providers cpu cuda
```

На 6 GB под Windows обучение падало с CUDA OOM на границе эпох при 2-4 GB свободной памяти:
ultralytics после валидации зовет `torch.cuda.empty_cache()`, а под WDDM память потом не всегда
выделяется заново. Помогли `--keep-cuda-cache` (отключает этот сброс) и batch 8 вместо 12.
Экспорт статический (1x3x640x640, opset 17), без NMS в графе: постобработка на numpy в
`firewatch/ops.py`, имена классов и размер входа берутся из метаданных ONNX.

## Запуск

```bash
firewatch detect path/to/images -o runs/detect   # картинки с разметкой
cp configs/cameras.example.yaml configs/cameras.yaml
cp .env.example .env                             # токен бота, chat_id, логины камер
firewatch run -c configs/cameras.yaml
```

Источник камеры - путь к файлу, RTSP/HTTP URL или номер веб-камеры. `${VAR}` и
`${VAR:-default}` подставляются из окружения и `.env`. `camera_defaults` применяются ко всем
камерам, пороги объединяются по классам, `smoke: null` у камеры отключает на ней дым.

Модель смотрит каждый `process_every`-й кадр; живой поток читается в отдельном потоке, и
берется самый свежий кадр, так что медленный инференс не копит задержку. Камера поднимает
тревогу, если класс прошел порог хотя бы в `min_hits` из последних `window` обработанных кадров:
так отсекаются блики и мигалки. Потом камера молчит `cooldown_s` секунд. Оборванный RTSP
переоткрывается через `reconnect_delay_s`.

В Telegram уходит снимок с рамками и подпись `[warehouse] fire 0.82, smoke 0.61` со временем
во второй строке. Если снимок не ушел, отправляется текст. Снимки сохраняются в `events/`.
`chat_id` можно узнать у @userinfobot.

## Тесты

`pytest` и `ruff check .`. Обученная модель не нужна: для детектора собирается крошечный
ONNX-граф с заранее известным выходом, запросы к Telegram подменяются `httpx.MockTransport`.

## Лицензия

AGPL-3.0, см. [LICENSE](LICENSE).

Датасет D-Fire (CC0 1.0): de Venâncio, Lisboa, Barbosa, Neural Computing and Applications, 2022.
