# Airline Satisfaction — Kaggle Playground S6E10

Решение задачи бинарной классификации удовлетворённости пассажиров авиакомпании.
Метрика соревнования — ROC AUC. Лучший зафиксированный public leaderboard score:
**0.96225**.

В проекте исследованы LightGBM, XGBoost, CatBoost, RealMLP, TabPFN,
feature engineering на основе маршрутов, leakage-safe target encoding,
пятифолдовый OOF-стекинг и ранговые ансамбли.

Подробное описание экспериментов, итоговой формулы, ограничений,
воспроизведения и готовые формулировки для резюме находятся в
[FINAL_SOLUTION.md](FINAL_SOLUTION.md).

Основные материалы:

- `notebooks/` — последовательность экспериментов;
- `scripts/` — анализ новой библиотеки из 112 моделей и построение финальных кандидатов;
- `artifacts/` — OOF/test-предсказания и метрики;
- `submissions/` — проверенные Kaggle submissions.

Финально выбранные файлы:

1. `xwc_v10_exact_submission.csv` — основной вариант, public ROC AUC 0.96225;
2. `xwc_v10_plus_new112_delta05.csv` — диверсифицированный вариант с тем же отображаемым public score.
