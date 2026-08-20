# recsys-hm

Двухэтапная (Retrieval → Ranking) рекомендательная система на датасете
H&M Personalized Fashion Recommendations. 

## Архитектура

```
Retrieval (Stage 1)              Ranking (Stage 2, обучен, не в проде)
─────────────────────            ────────────────────────────────────
Implicit ALS                     CatBoostRanker (YetiRank)
  top-100 кандидатов на юзера       переранжирование top-100 в top-10
  ↓ (холодный юзер)
Popularity fallback
  top-100 глобально популярных
```

**Production-модель — ALS + Popularity fallback (Фаза 6).**
CatBoostRanker обучен и полностью задокументирован (Фаза 8-9), но
сознательно не подключён к сервису: он не превосходит retrieval-этап по
качеству и активно вредит на cold-сегменте (подробности ниже).

## Результаты

Метрики на valid (все 187 783 юзера):

| Модель | Recall@10 | Recall@100 | NDCG@100 |
|---|---|---|---|
| Popularity baseline | 0.0097 | 0.0435 | 0.0160 |
| **ALS + fallback (прод)** | **0.0110** | **0.0443** | **0.0170** |
| CatBoostRanker | 0.0075 | 0.0443 | 0.0146 |

Сегментированная оценка (Фаза 9, cold = не встречались в train):

| Сегмент | Popularity Recall@10 | ALS+fallback Recall@10 | CatBoostRanker Recall@10 |
|---|---|---|---|
| Cold (8.3% юзеров) | 0.0096 | 0.0096 (= Popularity, ожидаемо) | **0.0026** |
| Warm | 0.0097 | **0.0111** (+14.6%) | 0.0079 |

Весь выигрыш ALS над Popularity сосредоточен в warm-сегменте — там, где
у ALS вообще есть история юзера. На cold-сегменте CatBoostRanker не
просто нейтрален, а активно вредит: переранжирует уже готовый
Popularity fallback хуже, чем если бы вообще его не трогал (тот же
набор кандидатов, но на 3.7x худший порядок внутри него) — модель ни
разу не видела cold-групп при обучении и лишена персонализированных
признаков на инференсе для этого сегмента. Полный разбор — в
`Доклад_recsys-hm.md`, разделы про Фазу 8 и Фазу 9.

## Структура проекта

```
src/
├── config.py               # централизованная загрузка configs/config.yaml
├── data/
│   ├── loader.py            # CSV -> Parquet ETL, customer_id -> customer_idx маппинг
│   └── split.py              # temporal train/valid/test split, фильтрация аномалий
├── recommenders/
│   ├── popularity.py         # непесонализированный baseline (Фаза 4)
│   └── als.py                 # Implicit ALS candidate generation (Фаза 5)
├── features/
│   └── builder.py            # feature engineering для ranker'а (Фаза 7)
├── ranking/
│   └── ranker.py              # CatBoostRanker train/inference (Фаза 8)
└── evaluation/
    ├── metrics.py              # Recall@K, NDCG@K, evaluate_by_segment
    ├── run_als_eval.py          # Фаза 6: ALS+fallback vs Popularity
    └── run_segmented_eval.py    # Фаза 9: сегментированная оценка cold/warm

service/
├── app.py                  # FastAPI: GET /recommend, GET /health
└── schemas.py               # Pydantic request/response контракты

tests/                      # pytest, 36 тестов (Фаза 11)
```

## Запуск

### Локально

```bash
pip install -e .

# ETL: data/raw/*.csv -> data/processed/*.parquet
python -m src.data.loader
python -m src.data.split

# сервис (обучает ALS+Popularity при старте, ~2 минуты)
uvicorn service.app:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`

### Docker

`data/` не входит в образ (в `.gitignore`, не часть кода) — пробрасывается
томом при запуске. Данные (`data/raw/*.csv`) нужно подготовить заранее
локальным ETL или скопировать уже готовые `data/processed/*.parquet`.

```bash
docker build -t recsys-hm .
docker run -p 8000:8000 -v $(pwd)/data:/app/data recsys-hm
```

### Пример запроса

```bash
curl "http://127.0.0.1:8000/recommend?user_id=<hex_customer_id>&top_k=10"
```

```json
{
  "customer_id": "<hex_customer_id>",
  "recommendations": [706016001, 706016002, 372860001, ...],
  "source": "als"
}
```

`source` явно указывает, персонализирован ли результат (`"als"`) или это
Popularity fallback для холодного/неизвестного юзера
(`"popularity_fallback"`).

## Ключевые архитектурные решения

- **Temporal split**, не random — защита от Temporal Leakage; все
  агрегаты (популярность, ALS-матрица) считаются строго по train.
- **Popularity ранжируется по уникальным покупателям**, не по числу
  транзакций — сигнал охвата аудитории, не объёма продаж (см. EDA,
  раздел про дубликаты транзакций).
- **Исключение уже купленного** реализовано вручную поверх сырых
  рекомендаций ALS, не через встроенный флаг библиотеки — для
  прозрачного логирования и тестируемости (Фаза 11:
  `test_excludes_already_purchased_items`).
- **Честное сравнение моделей на офлайн-оценке** — холодные юзеры
  получают Popularity fallback вместо исключения из усреднения; иначе
  Recall@K для ALS считался бы только на более лёгком подмножестве
  юзеров, чем у Popularity.
- **CatBoostRanker не в проде** — не улучшение, отклонённое за
  недостатком времени, а зафиксированное решение по итогам диагностики
  (4 гипотезы проверены и отвергнуты в Фазе 8, количественно
  подтверждено в Фазе 9). Модель, файлы и находки полностью
  задокументированы для истории эксперимента.

## Тесты

```bash
pytest tests/ -v
```

36 тестов: математика метрик на контролируемых примерах
(`test_metrics.py`), `PopularityRecommender`/`ALSCandidateGenerator` на
реально обученной (маленькой синтетической) ALS-модели, не моках
(`test_recommenders.py`), интеграционные тесты FastAPI-эндпоинтов через
`TestClient` (`test_api.py`).
