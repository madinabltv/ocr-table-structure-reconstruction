# Восстановление структуры таблиц после OCR

Проект восстанавливает логическую структуру русскоязычных таблиц по тексту и координатам OCR-фрагментов. Для кандидатных пар фрагментов определяются отношения `SAME_CELL`, `RIGHT`, `BELOW` и `NO_RELATION`, после чего строятся строки, столбцы и объединённые ячейки.

Поддерживаются три режима реконструкции:

- `grid` — таблица с полной сеткой;
- `partial_grid` — таблица с частично видимыми границами;
- `hybrid` — восстановление по отношениям и геометрии при неявной сетке.

## Структура проекта

- `src/` — OCR, построение признаков, обучение, предсказание, реконструкция и оценка;
- `tests/` — модульные тесты и небольшой пример формата SciTSR;
- `data/images/` — изображения русскоязычных таблиц;
- `data/ocr/` — сохранённые результаты OCR;
- `annotations/` — разметка логических ячеек;
- `experiments/` — манифесты экспериментальных выборок;
- `results/` — компактные итоговые таблицы и графики;
- `examples/` — небольшой пример отношений между фрагментами.

Большие исходные файлы SciTSR, производные JSONL, обученные модели и промежуточные результаты в репозиторий не включены.

## Требования

- Python 3.10 или новее;
- Tesseract OCR;
- языковые пакеты Tesseract для русского и английского языков.

Создание окружения:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

В Windows команда активации окружения имеет вид:

```powershell
.venv\Scripts\activate
```

## Проверка проекта

Тесты не требуют `pytest` и запускаются стандартными средствами Python:

```bash
python -m unittest discover -s tests -v
```

## Пример реконструкции

Ниже приведён пример на уже подготовленном OCR-файле. Все команды выполняются из корня проекта.

```bash
mkdir -p outputs/demo

python src/build_relation_baseline.py \
  --input data/ocr/table_09_tax_elements.json \
  --output outputs/demo/relations.json

python src/auto_reconstruct_table.py \
  --image data/images/table_09_tax_elements.png \
  --ocr data/ocr/table_09_tax_elements.json \
  --relations outputs/demo/relations.json \
  --output outputs/demo/structure.json \
  --diagnostics-output outputs/demo/diagnostics.json \
  --preview-output outputs/demo/preview.png

python src/evaluate_structure.py \
  --prediction outputs/demo/structure.json \
  --ground-truth annotations/table_09_tax_elements_cells_ground_truth.json \
  --output outputs/demo/evaluation.json
```

Для подготовки иллюстрации с исходным изображением, OCR-прямоугольниками и восстановленными ячейками:

```bash
python src/make_reconstruction_figure.py \
  --image data/images/table_09_tax_elements.png \
  --ocr data/ocr/table_09_tax_elements.json \
  --structure outputs/demo/structure.json \
  --output outputs/demo/reconstruction_example.png \
  --show-text
```

## Работа с SciTSR

Набор [SciTSR](https://github.com/Academic-Hammer/SciTSR) необходимо загрузить отдельно. Его исходные файлы не следует добавлять в Git. После загрузки укажите каталоги `chunk` и `rel` явно:

```bash
python src/convert_scitsr.py \
  --chunk-dir /path/to/SciTSR/train/chunk \
  --relation-dir /path/to/SciTSR/train/rel \
  --output data/processed/scitsr_train.jsonl \
  --same-cell-rate 0.2
```

Пример обучения геометрического классификатора:

```bash
mkdir -p outputs/models outputs/training

python src/train_geometric_classifier.py \
  --input data/processed/scitsr_train.jsonl \
  --model-output outputs/models/geometric.joblib \
  --report-output outputs/training/geometric_report.json \
  --feature-set geometry \
  --classifier logreg
```

Параметры конкретных экспериментальных запусков зафиксированы в скриптах и манифестах проекта.

## Данные и воспроизводимость

Перед публикацией изображений и разметки убедитесь, что условия их источников допускают распространение. Для SciTSR следует сохранить ссылку на исходный набор и его лицензию. Случайное состояние обучаемых моделей фиксируется, а разделение данных выполняется по документам.

Репозиторий пока не содержит файла лицензии. Перед публичной публикацией выберите лицензию на собственный код отдельно; лицензия кода не заменяет лицензии используемых наборов данных и моделей.
