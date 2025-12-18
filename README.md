# firewatch

Детекция огня и дыма на видеопотоках с IP-камер и из видеофайлов. Модель YOLO11s,
обученная на открытом датасете D-Fire, на инференсе работает через onnxruntime без PyTorch.
При устойчивом срабатывании отправляет в Telegram кадр с разметкой и сохраняет снимок.

Пока в работе: модель обучается, результатов и весов еще нет.

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
python scripts/train.py --resume runs/yolo11s-dfire/weights/last.pt
python scripts/evaluate.py runs/yolo11s-dfire/weights/best.pt --split test --out docs/test_metrics_pt.json
```

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
