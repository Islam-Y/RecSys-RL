# T-Lab 2026 - RecSys RL

Второе задание: GPT-2 для next-item recommendation, модель вознаграждения и RL-дообучение. Первая серия завершена фиксированной оценкой на final test. Дополнительно проверены смешанная цель и явный учёт повторных прослушиваний; каждый новый подбор отделён от предыдущих открытых выборок.

## Итоговые материалы

- [Отчёт для чтения](Отчёт.md) и [PDF, 6 страниц](output/pdf/recsys-rl-report.pdf).
- [Условие](Задание.md), [подробное исследование](Решение.md), [журнал работы](История-исследования.md).
- [Final test: результат](experiments/final-evaluation-result.json), [протокол](experiments/final-evaluation-protocol.md), [аудит и ресурсы](experiments/final-evaluation-execution.json).
- [RL-пилот: результат](experiments/rl-pilot-result.json), [протокол](experiments/rl-pilot-protocol.md), [ресурсы](experiments/rl-pilot-execution.json).

## Последняя серия: явный учёт повторов - 15 октября

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

Текущий запуск уже завершён: для сверки используйте `--audit`. Новые запуски требуют пустых `--output-dir`/`--holdout-dir` и локальных артефактов предыдущих этапов. Для исторического аудита нужен код записанного в отчёте коммита: дополнительные исходники меняют полный source SHA. Ни одна открытая оценочная группа не считается независимым подтверждением дальнейшего подбора.

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

Нужны локальные данные и checkpoints с хешами из отчётов. В текущем рабочем каталоге результат уже существует, поэтому используется аудит:

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

Python 3.14.7; зависимости зафиксированы в `requirements-lock.txt`: PyTorch 2.14.1, Transformers 5.19.0, NumPy 2.5.4, DuckDB 1.5.6. Для указанного NumPy нужен Python >= 3.12.

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
Отчёт.md                 # Итоговый отчёт, источник PDF
Решение.md               # Подробное описание этапов и результатов
История-исследования.md  # Хронологический журнал решений
output/pdf/              # Итоговый PDF
```

`data/`, `.venv/`, `outputs/` и временные рендеры исключены из Git. Итоговый PDF включён как материал для подачи. Исторические результаты сохранены: [первый baseline](experiments/baseline-result.json), [размеры](experiments/baseline-capacity-result.json), [промежуточная модель](experiments/baseline-medium-result.json), [настройки](experiments/small-tuning-result.json), [reward model](experiments/reward-result.json), [проверка reward-сигнала](experiments/reward-signal-result.json).

Основные пункты задания выполнены: baseline, reward model, RL-дообучение, сравнение релевантности/поведения, несколько reward, rollout и обсуждение долгосрочных целей. Дальнейшие расширения - более широкий каталог, сильнее проверенный reward и моделирование реальных переходов - остаются отдельным исследованием.
