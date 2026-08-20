# recsys-hm — Dockerfile (Фаза 12 Master Execution Plan)
#
# Собирает и запускает FastAPI-сервис (service/app.py), который при
# старте (lifespan) обучает Popularity + ALS на train.parquet и держит
# их в памяти процесса — см. докстринг service/app.py, пункт 1 (обучение
# при старте, без сохранённых на диск ALS-артефактов, ~2 минуты на
# полном train).
#
# data/ НЕ копируется в образ. Она в .gitignore и не является частью
# кода проекта, а её объём (сырые CSV ~700MB + processed parquet)
# неоправданно раздул бы образ и не имеет смысла запекать в него —
# данные пробрасываются снаружи томом при запуске:
#
#   docker build -t recsys-hm .
#   docker run -p 8000:8000 -v $(pwd)/data:/app/data recsys-hm
#
# models/catboost_ranker.cbm по той же причине не копируется — ranker
# не идёт в прод (зафиксированное решение Фаз 8-9, см. Доклад,
# раздел 13), service/app.py его не читает вообще.

FROM python:3.10-slim

# requires-python в pyproject.toml зафиксирован как >=3.10,<3.11 —
# implicit/catboost на момент Фазы 5/8 собирались и проверялись именно
# под 3.10, поэтому базовый образ жёстко закреплён на этой версии, а не
# взят как "latest slim".

WORKDIR /app

# Системные зависимости для сборки implicit/scipy (нужны заголовки C для
# компиляции нативных расширений на некоторых платформах) и curl —
# используется в HEALTHCHECK ниже, в самом слим-образе его нет.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала — только манифест зависимостей, чтобы pip install кешировался
# отдельным слоем и не переустанавливался при каждой правке кода в src/
# или service/ (самый частый вид изменений при разработке).
COPY pyproject.toml README.md ./

# Заглушки для setuptools.packages.find (include = ["src*", "service*"]
# в pyproject.toml) — сборка пакета в этом слое падает без реальных
# директорий src/ и service/, а копировать сам код сюда преждевременно
# (сломало бы кеш слоя при любой правке). Пустые __init__.py достаточно
# для успешной установки зависимостей; реальный код перезапишет их на
# следующем шаге.
RUN mkdir -p src service && touch src/__init__.py service/__init__.py

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Теперь код — этот слой пересобирается на каждую правку src/service,
# но предыдущий (тяжёлый pip install) остаётся закешированным.
COPY src/ ./src/
COPY service/ ./service/
COPY configs/ ./configs/

# models/ копируется, а не монтируется томом: единственный артефакт там —
# catboost_ranker.cbm, который сервис не использует (см. докстринг выше);
# каталог нужен только чтобы CONFIG.paths.models существовал, если
# какой-то будущий код в него обратится. Пустой .gitkeep — не веса модели.
COPY models/.gitkeep ./models/.gitkeep

EXPOSE 8000

# Проверяет реальную готовность сервиса — /health отдаёт 200 только
# после того, как lifespan закончил обучение ALS+Popularity (см.
# service/app.py: model_state=None -> 503 до этого момента), а не
# просто "процесс запущен".
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
