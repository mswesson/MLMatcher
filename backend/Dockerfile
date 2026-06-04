# ML-стек (CatBoost, numpy, pandas, scikit-learn) требует бинарных wheel'ов.
# Берём Python 3.12 — для него гарантированно есть manylinux-колёса всех зависимостей.
FROM python:3.12-slim

# libgomp нужен CatBoost/scikit-learn для OpenMP-параллелизма.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

EXPOSE 8000

# In-memory стор задач требует ОДНОГО воркера (см. ТЗ).
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
