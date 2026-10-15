# T-Lab 2026 - RecSys RL

Второе задание: GPT-2 для next-item recommendation, модель вознаграждения и RL-дообучение. Проверены несколько reward, смешанная цель, явный учёт повторов и обучение ранжировщика на расширенных данных. Последняя серия подтвердила улучшение выбора кандидатов; последующее RL повысило прогнозируемый reward, но не подтвердило сохранение NDCG в пределах 1%.

## Итоговые материалы

- **[Актуальный сводный отчёт](Сводный-отчёт.md)** и [PDF всех серий](output/pdf/recsys-rl-ranking-policy.pdf).
- [Предыдущий сводный отчёт](Итоговый-отчёт.md) и [его PDF](output/pdf/recsys-rl-final.pdf) сохранены как снимок до расширения ranking/RL.
- [Отчёт первой серии](Отчёт.md) и [её PDF, 6 страниц](output/pdf/recsys-rl-report.pdf).
- [Условие](Задание.md), [подробное исследование](Решение.md), [журнал работы](История-исследования.md).
- [Final test: результат](experiments/final-evaluation-result.json), [протокол](experiments/final-evaluation-protocol.md), [аудит и ресурсы](experiments/final-evaluation-execution.json).
- [RL-пилот: результат](experiments/rl-pilot-result.json), [протокол](experiments/rl-pilot-protocol.md), [ресурсы](experiments/rl-pilot-execution.json).

## Последние серии: обучение ранжированию и повторный RL - 15 октября

**Выбор кандидатов улучшен на независимой группе новых пользователей.** CatBoostRanker на расширенных данных получил query-macro accuracy **58.42%** против **53.52%** у контроля прошлых откликов на кандидата. Парная разница +4.90 п.п., 95% user-cluster интервал [+1.51; +8.19] п.п. Проверены 2 524 сравнения с разрывом долей прослушивания >= 0.10 в 520 запросах от 173 пользователей. Эта метрика измеряет порядок двух наблюдавшихся кандидатов; это не NDCG и не CTR.

Сравнены числовая MLP и CatBoostRanker на одинаковых 16 признаках, малой и расширенной выборках. Расширение: 1 743 → 46 373 обучающих сравнения, 427 → 9 881 информативный запрос. Все сравнения — реально наблюдавшиеся отклики, искусственных негативов нет. Четыре рецепта и два подтверждающих seed; выбран expanded CatBoost seed 42, 320 деревьев. Калибровка в прогноз доли прослушивания выполнена отдельно, после выбора ранжировщика.

Новый reward прошёл критерии допуска к RL. Reference — ранее выбранная GPT-2 + recency; шесть RL-обучений и 16 greedy/sampled rollout. Выбрана политика с KL = 0.1 и штрафом соседнего повтора 0.1, прошедшая validation-проверки на seeds 42/43/44. Финальный резерв открыт после фиксации выбора: 5 000 запросов, 739 пользователей.

| Последний final reserve | NDCG@10 | Recall@10 | Calibrated top-10 reward |
|---|---:|---:|---:|
| GPT-2 + recency reference | 0.09570 | 0.15246 | 0.66548 |
| Выбранная RL | 0.09517 | 0.15100 | 0.66747 |
| Запрет соседнего повтора | 0.08166 | 0.13386 | 0.66588 |
| Недавние треки + популярность | 0.09583 | 0.15416 | 0.67102 |

RL: reward **+0.30%**, интервал абсолютной разницы [+0.001837; +0.002156]. NDCG **−0.56%** по точечной оценке, интервал разницы [−0.001896; +0.000810]. Допустимая граница −0.000957 не пройдена: **сохранение NDCG в пределах 1% не подтверждено, общий критерий RL не выполнен**. Это не доказательство потери более 1%; данных недостаточно для заданного подтверждения. Преимущество над recent-items и пользовательский эффект не установлены.

В sampled-пятёрке выбранная политика дала 4.203 уникального трека против 4.121 у reference; повтор на шаге 5 — 17.3% против 17.1%. Greedy-зацикливание осталось. Жёсткий запрет убрал соседние повторы, но снизил final NDCG на 14.68%. Во время rollout история наблюдавшихся откликов для reward фиксирована; сгенерированные ID не считаются реальными прослушиваниями.

[Ranking: протокол](experiments/ranking-protocol.md), [результат](experiments/ranking-result.json), [независимое подтверждение](experiments/ranking-confirmation-result.json), [подготовка](experiments/ranking-preparation.json), [аудит/ресурсы](experiments/ranking-execution.json). [RL: протокол](experiments/ranked-policy-protocol.md), [конфигурация](configs/ranked-policy.json), [результат](experiments/ranked-policy-result.json), [final](experiments/ranked-policy-final-result.json), [подготовка](experiments/ranked-policy-preparation.json), [аудит/ресурсы](experiments/ranked-policy-execution.json).

CatBoost **1.2.10**, последняя стабильная версия на дату запуска, проверена по [официальному PyPI](https://pypi.org/project/catboost/). Зависимости расширения зафиксированы отдельно в `requirements-ranking-lock.txt`; прежний lock сохранён. Ranking-серия заняла 57.46 с на CPU, RL-серия — 127.74 с на Apple M4/MPS. Это время малого прототипа, не оценка производственной пропускной способности.

```bash
source .venv/bin/activate
python -m pip install -r requirements-ranking-lock.txt
python -m recsys_rl.ranked_policy_experiment --device auto --audit
```

Аудит последней RL-серии воспроизвёл raw-истории, full-catalog reward, шесть checkpoint, 16 rollout, выбор и final; разница policy forward 0, reward-кэша с прямым расчётом < 3e-8. Автоматические тесты не добавлялись и не запускались.

Команды исходных запусков:

```bash
python -m recsys_rl.ranking_experiment --prepare-only
python -m recsys_rl.ranking_experiment --resume
python -m recsys_rl.ranking_experiment --audit
python -m recsys_rl.ranked_policy_experiment --device auto --prepare-only
python -m recsys_rl.ranked_policy_experiment --device auto --resume
python -m recsys_rl.ranked_policy_experiment --device auto --audit
```

Нужны локальные данные/checkpoints прежних этапов и пустые каталоги для новых запусков. Исторический ranking-аудит требует исходников `669bec6`; текущий RL-аудит — замороженного кода `3de7c53`. После ranking/RL открыты и confirmation, и final reserve: эти группы больше не являются независимым подтверждением будущего подбора. Флаги `final_reserve_opened: false` в старых подготовительных JSON описывают состояние на момент их записи.

## Предыдущая серия: фиксированная смесь и история откликов - 15 октября

**История откликов улучшила прогноз доли прослушивания, но не дала достаточно надёжного выбора кандидатов для нового RL.** На 5 868 наблюдавшихся парах от 383 новых development-пользователей MSE MLP 0.12524 против 0.19679 прежнего hybrid (−36.36%) и 0.14341 средней прошлых откликов пользователя (−12.67%). Development используется для выбора рецепта и не является final test.

Три обучения: линейная поправка, MLP с embeddings, pairwise MLP. При разрыве меток >= 0.10 последний получил query-macro accuracy 47.58% [39.48%; 55.34%], лучший MSE-вариант — 44.20%. Поддержка: 120 запросов, 81 пользователь. Ни один не прошёл заранее заданный порог 55%; RL на этих моделях пропущено, пороги не менялись. Тогда отдельные 1 000 final-пользователей вне прежних 3 500, 5 000 запросов, оставались без прогнозов/response-меток. Этот резерв открыт только в последней серии с новым ранжировщиком.

Контроль смеси на уже открытой recency-группе: обученные 10.804% дали NDCG 0.10133, фиксированные 10% — 0.10018, начальная доля 11.920% — 0.10161. Это последующая диагностика, не независимое подтверждение. Полезность обучения доли отдельно от recency-механизма не установлена; предыдущий выбор задним числом не заменён.

[Протокол](experiments/reward-history-protocol.md), [конфигурация](configs/reward-history.json), [результаты](experiments/reward-history-result.json), [подготовка групп](experiments/reward-history-preparation.json), [аудит, ресурсы и пояснение весов bootstrap](experiments/reward-history-execution.json).

```bash
python -m recsys_rl.history_reward_experiment --device auto --prepare-only
python -m recsys_rl.history_reward_experiment --device auto --resume
python -m recsys_rl.history_reward_experiment --device auto --audit
```

Сохранённый запуск проверяется `--audit` с кодом его этапа. Независимое повторение требует локальных исходных артефактов и пустого output; новый путь подготовленных групп задаётся `new_data_dir` в отдельной конфигурации. Проверены raw-истории с отсечением T'<T, все три checkpoint, метрики/выбор и непересечение групп; максимальная разница forward 0. Код серии — `bc91ff4ba21887913a74cfb83b85674eee537371`.

## Предыдущая серия: явный учёт повторов - 15 октября

Смесь замороженной GPT-2 с недавними треками дала на новой группе пользователей **NDCG@10 +23.73%, Recall@10 +26.55% относительно GPT-2**, интервалы разниц выше нуля. Добавлен один обучаемый параметр: доля recency 10.8%, параметр экспоненциального затухания tau = 16 событий. Восемь обучений, все 59 888 train-запросов; исходные веса GPT-2 сохранены.

| Метод | Новый holdout NDCG@10 | Recall@10 |
|---|---:|---:|
| GPT-2 | 0.08190 | 0.12501 |
| Недавние треки + популярность | 0.09983 | 0.15547 |
| GPT-2 + recency | **0.10133** | **0.15820** |

**Победа над recent-items не подтверждена:** NDCG +1.51% по точечной оценке, интервал разницы [-0.00246; +0.00545]. Основной критерий требовал победы над обоими контролями, поэтому полностью не выполнен. Прирост связан с повторными прослушиваниями; на целях, новых для истории, смесь хуже GPT-2. Sampled-повторы на пятом шаге выросли с 16.6% до 17.1%; greedy-зацикливание сохранилось.

Новый holdout: 5 000 запросов, 753 пользователя; полная группа 1 000 пользователей исключает обе прежние популяции по 1 000. Когорта запечатана до обучения, оценка открыта после фиксации выбора. Это улучшение baseline; reward model/RL не переобучены, пользовательский эффект не измерен.

[Описание серии](Учёт-повторов.md), [PDF](output/pdf/recsys-rl-recency.pdf), [протокол](experiments/recency-protocol.md), [результат](experiments/recency-result.json), [holdout](experiments/recency-holdout-result.json), [подготовка](experiments/recency-holdout-preparation.json), [аудит и ресурсы](experiments/recency-execution.json).

```bash
python -m recsys_rl.recency_experiment --device auto --prepare-only
python -m recsys_rl.recency_experiment --device auto --resume
python -m recsys_rl.recency_experiment --device auto --audit
```

Серия завершена: для сверки используйте `--audit` с кодом `5c8158e`. Новые запуски требуют пустых `--output-dir`/`--holdout-dir` и локальных артефактов предыдущих этапов. Для исторического аудита нужен код записанного в отчёте коммита: дополнительные исходники меняют полный source SHA. Ни одна открытая оценочная группа не считается независимым подтверждением дальнейшего подбора.

## Предыдущая дополнительная серия: смешанная цель - 15 октября

После завершения первой серии проверено смешивание RL с next-item loss: шесть обучений и 14 rollout. **Новый смешанный вариант не принят:** NDCG снизился на 3.19-3.60%, превысив заранее заданный допуск 1%. Повторы sampled-пятёрки уменьшились, но критерий релевантности не выполнен.

Для итоговой проверки подготовлены 1 000 новых пользователей, полностью исключённых из первоначальной популяции; 5 000 запросов, 751 пользователь в запросах, 5 694 цели. После фиксации решения оценены заранее заданные контроли. Сильный KL дал hybrid reward +0.36% и NDCG +0.44% по точечной оценке; интервал NDCG включает ноль. Недавние треки снова выше нейросетевых политик. Смешанные варианты на новом holdout не оценивались, поскольку не прошли validation-отбор.

Первая серия и её PDF сохранены. [Описание дополнения](Дополнение.md), [PDF, 3 страницы](output/pdf/recsys-rl-mixed-objective.pdf), [протокол](experiments/mixed-objective-protocol.md), [результат](experiments/mixed-objective-result.json), [новая популяция](experiments/mixed-objective-holdout-result.json), [подготовка holdout](experiments/mixed-holdout-preparation.json), [аудит и ресурсы](experiments/mixed-objective-execution.json).

```bash
python -m recsys_rl.mixed_experiment --device auto --prepare-only
python -m recsys_rl.mixed_experiment --device auto --resume
python -m recsys_rl.mixed_experiment --device auto --audit
```

Команды используют локальные артефакты первой серии. Старый test не переоценивался; новая выборка теперь также открыта и не может служить независимым подтверждением последующего подбора.

## Результат первой серии

5 000 test-запросов, 755 пользователей, 5 793 цели; одинаковый каталог из 2 000 треков. Hybrid reward - средний прогноз top-10 с равными весами позиций.

| Политика | NDCG@10 | Recall@10 | Hybrid reward |
|---|---:|---:|---:|
| Исходная GPT-2 | 0.09121 | 0.14119 | 0.64693 |
| Выбранная RL, KL = 0.1 | 0.08889 | 0.14153 | **0.65979** |
| Дополнительная RL, KL = 1 | 0.09137 | 0.14146 | 0.64871 |
| Train-популярность | 0.01179 | 0.02614 | 0.64904 |
| Недавние треки + популярность | **0.10026** | **0.15626** | 0.65237 |

Выбранная по validation политика повысила hybrid reward на **1.99%**, NDCG снизился на **2.55%** по точечной оценке. Парный user-cluster интервал разницы NDCG [-0.004946; +0.000226] включает ноль; это не доказывает эквивалентность качества. Сильный KL дал меньший прирост reward (+0.28%) при близких ranking-метриках; он заранее задан как дополнительное сравнение, основной выбор по test не менялся.

В validation-пилоте отсутствие KL привело к коллапсу: NDCG -61.85%, десять наиболее выдаваемых треков заняли 72.91% top-10 выдачи. Greedy-rollout почти зацикливался уже у baseline. На пятом sampled-шаге повтор предыдущего трека: baseline 16.6%, выбранная RL 25.2%, без KL 96.2%.

**Reward - прогноз, не доказанная польза пользователю.** Удовлетворённость, причинный прирост дослушивания и удержание не измерены. Офлайн выявлены компромисс и ошибки прокси; в первой серии ни одна обученная политика не превзошла контроль недавних треков по next-item качеству.

## Метод

- Данные: [Yambda-50M](https://huggingface.co/datasets/yandex/yambda), 1 000 пользователей, каталог по train. История - BOS и до 64 прошлых прослушиваний; 61.77% train-контекста представлено UNK.
- Временные границы: train до `19983235`, validation до `23121935`, test до `26000000` секунд. Все события текущего user/T исключены из контекста и оцениваются множеством целей.
- GPT-2: 2 слоя, ширина 128, 4 головы, 661 504 параметра, собственные ID-токены. Сравнены три размера и восемь рецептов; выбран learning rate 0.0001, reference seed 42, эпоха 26.
- Reward model: embeddings/MLP, 202 593 параметра; прогноз доли прослушивания. На test raw MSE 0.18133, персональная средняя 0.17564, фиксированная hybrid-смесь 0.17121. Hybrid улучшает MSE, но ухудшает MAE относительно персональной средней.
- RL: точное ожидаемое вознаграждение по каталогу со штрафом KL; пять эпох на 4 000 поздних train-историях. Шесть вариантов reward/регуляризации и два подтверждающих seed: восемь обучений. Reward model и reference заморожены.
- Задача обучения одношаговая. Rollout пяти объектов использует синтетическое добавление выбранного трека; реальные переходы и многошаговый return не обучались.

## Запуск и аудит сохранённого final test

Нужны локальные данные/checkpoints с хешами из отчётов и код исторической оценки `c97ac8f`. Сохранённый результат проверяется аудитом:

```bash
source .venv/bin/activate
python -m recsys_rl.final_evaluation --device auto --audit
```

Для независимого повторения оценки тех же весов задаётся новый `--output-dir`. Это пересчёт фиксированных моделей, а не повод подбирать настройки по test:

```bash
python -m recsys_rl.final_evaluation --device auto --output-dir outputs/final-evaluation-repeat
python -m recsys_rl.final_evaluation --device auto --output-dir outputs/final-evaluation-repeat --audit
```

Аудит повторно присоединил метки из raw, воспроизвёл logits трёх политик и reward полного каталога с максимальной разницей 0, пересчитал выдачи, метрики, интервалы и контроли. Final evaluation заняла 16.49 с, аудит - 14.13 с на Apple M4/MPS. Код этой оценки: `c97ac8f0bf4656b24eeb74aa161f6b6c853d7594`.

## Полное повторение с обучением

Используйте **отдельный клон и пустые каталоги данных/запусков**. Checkpoints и сырые данные в Git не включены. Опубликованные отчёты относятся к исходному запуску; между этапами нужно подставлять отчёты собственного повторения. Пути и хеши manifest привязаны к месту подготовки, поэтому перенос оригинальных JSON без соответствующих артефактов не заменяет подготовку.

Python 3.14.7; зависимости исходной серии зафиксированы в `requirements-lock.txt`: PyTorch 2.14.1, Transformers 5.19.0, NumPy 2.5.4, DuckDB 1.5.6. Для указанного NumPy нужен Python >= 3.12. Для ranking/RL-расширения используется `requirements-ranking-lock.txt` с CatBoost 1.2.10.

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
mkdir -p data/raw
curl -L --fail --output data/raw/yambda-50m-multi_event.parquet https://huggingface.co/datasets/yandex/yambda/resolve/main/flat/50m/multi_event.parquet
python -m recsys_rl prepare data/raw/yambda-50m-multi_event.parquet
python -m recsys_rl tune --device auto
cp outputs/small-tuning-v1/result.json experiments/small-tuning-result.json
python -m recsys_rl reward-prepare data/raw/yambda-50m-multi_event.parquet
python -m recsys_rl reward --device auto
cp outputs/reward-v1/summary.json experiments/reward-result.json
python -m recsys_rl reward-signal --device auto
cp outputs/reward-signal-v1/result.json experiments/reward-signal-result.json
python -m recsys_rl rl --device auto --prepare-only
python -m recsys_rl rl --device auto --resume
python -m recsys_rl rl-audit --device auto
cp outputs/policy-rl-pilot-v1/result.json experiments/rl-pilot-result.json
```

В final-конфигурации нужно заранее зафиксировать выбранный по validation checkpoint. В опубликованном запуске это raw-kl seed 42; если собственное повторение выбрало другой рецепт, конфигурацию/протокол необходимо согласовать **до** открытия его test. Для повторения текущей конфигурации после генерации собственных отчётов обновите привязку SHA пилота:

```bash
python - <<'PY'
from pathlib import Path
from recsys_rl.data import sha256, write_json
write_json(Path('experiments/rl-pilot-execution.json'), {
    'result_sha256': sha256(Path('experiments/rl-pilot-result.json')),
    'scope': 'Local reproduction; process accounting not collected.'
})
PY
python -m recsys_rl.final_evaluation --device auto
python -m recsys_rl.final_evaluation --device auto --audit
```

Это создаёт метаданные нового повторения; исходные измерения ресурсов ему не приписываются. Для их получения запускайте команды через `/usr/bin/time -l`. Повторения могут отличаться из-за устройства и численной точности. Непустые каталоги не перезаписываются, незавершённые запуски нужно сохранять отдельно. Исторические аудиты требуют совпадения кода с сохранённой версией этапа; добавление новых модулей меняет полный снимок исходников.

## Структура и предыдущие этапы

```text
configs/                 # Рецепты baseline, reward, RL и final evaluation
recsys_rl/               # Подготовка, обучение, оценка и аудит
experiments/             # Протоколы и агрегированные результаты
Сводный-отчёт.md          # Актуальный отчёт всех серий, источник PDF
Отчёт.md                 # Сохранённый отчёт исходной серии
Решение.md               # Подробное описание этапов и результатов
История-исследования.md  # Хронологический журнал решений
output/pdf/              # Актуальный и исторические PDF
```

`data/`, `.venv/`, `outputs/` и временные рендеры исключены из Git. Итоговый PDF включён как материал для подачи. Исторические результаты сохранены: [первый baseline](experiments/baseline-result.json), [размеры](experiments/baseline-capacity-result.json), [промежуточная модель](experiments/baseline-medium-result.json), [настройки](experiments/small-tuning-result.json), [reward model](experiments/reward-result.json), [проверка reward-сигнала](experiments/reward-signal-result.json).

Основные пункты задания выполнены: baseline, reward model, RL-дообучение, сравнение релевантности/поведения, несколько reward, rollout и обсуждение долгосрочных целей. Расширения дали подтверждённый сигнал ранжирования, но общий финальный критерий RL не выполнен. Более широкий каталог, другой источник данных и моделирование реальных переходов остаются отдельным исследованием.
