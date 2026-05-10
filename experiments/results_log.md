# Experiments Results Log — Concrete Strength GAN

## Summary Table

| Version | Val MAE | Val RMSE | Val MAPE | Val R² | PICP95 | GOST viol. | Notes |
|---------|---------|----------|----------|--------|--------|------------|-------|
| **v1** | 2.054 | 2.577 | 9.74% | 0.913 | 0.531 | 0/40 | Baseline |
| **v2** | **1.001** | **1.287** | **3.95%** | **0.978** | 0.625 | 0/40 | Bigger net, MAPE loss, more epochs |
| **v3** | 1.117 | 1.449 | 4.90% | 0.972 | **0.825** | 0/40 | period_weights, noise_dim=24, dropout=0.10, wider BNN |
| Δ (v3−v2) | +0.116 (+12%) | +0.162 (+13%) | +0.95pp | −0.006 | +0.200 ✓ | = | Day-28 R²: 0.719→0.782 ✓ |
| **v4** | **1.054** | **1.408** | 4.90% | **0.974** | 0.725 | 0/40 | +4 domain features: log(c/w), eff. w/c, agg/binder, plasticizer% |
| Δ (v4−v3) | −0.063 (−6%) | −0.041 (−3%) | =0 | +0.002 | −0.100 | = | Day-1 R²: 0.808→0.927 ✓; Day-7: 0.872→0.810 ✗ |
| **v5** | 1.216 | 1.546 | 5.53% | 0.969 | 0.750 | 0/40 | Config-only tuning from v4; heavy Day-28 weighting |
| Δ (v5−v4) | +0.162 (+15%) | +0.138 (+10%) | +0.63pp | −0.005 | +0.025 | = | Day-28 R²: 0.769→0.771 (flat), overall quality regressed |
| **v6** | **0.709** | **0.931** | **3.10%** | **0.989** | **0.988** | 0/40 | +Heteroscedastic uncertainty head + NLL loss |
| Δ (v6−v5) | −0.507 (−42%) | −0.615 (−40%) | −2.42pp | +0.020 | +0.238 | = | Day-28 R²: 0.771→0.856 ✓, conformal PICP95=0.944 |
| **v7** | 0.722 | 1.150 | 3.72% | 0.984 | 0.956 | 0/40 | Larger model + stronger uncertainty; best Day-28 and calibrated coverage |
| Δ (v7−v6) | +0.013 (+2%) | +0.219 (+24%) | +0.62pp | −0.005 | −0.032 | = | Day-28 R²: 0.856→0.911 ✓, PICP95 near target 0.95 |

---

## Per-Period Metrics (Validation)

| Version | Day-1 MAE | Day-1 R² | Day-3 MAE | Day-3 R² | Day-7 MAE | Day-7 R² | Day-28 MAE | Day-28 R² |
|---------|-----------|----------|-----------|----------|-----------|----------|------------|----------|
| **v1** | 1.639 | 0.682 | 2.392 | 0.464 | 2.108 | 0.580 | 2.079 | 0.441 |
| **v2** | **0.477** | **0.969** | **0.813** | **0.946** | **1.172** | **0.867** | 1.544 | 0.719 |
| **v3** | 1.096 | 0.808 | 0.958 | 0.917 | 1.101 | 0.872 | **1.313** | **0.782** |
| **v4** | **0.787** | **0.927** | **0.896** | **0.919** | 1.231 | 0.810 | **1.302** | 0.769 |
| Δ v4−v3 | −0.309 | **+0.119** | −0.062 | +0.002 | +0.130 | −0.062 | −0.011 | −0.013 |
| **v5** | 0.980 | 0.865 | 1.327 | 0.853 | 1.272 | 0.833 | 1.285 | 0.771 |
| **v6** | **0.708** | **0.941** | **0.477** | **0.982** | **0.708** | **0.936** | **0.945** | **0.856** |
| **v7** | 0.833 | 0.871 | 0.640 | 0.928 | 0.774 | 0.909 | **0.638** | **0.911** |

---

## Discriminator

| Version | Realness mean | Realness std |
|---------|--------------|--------------|
| **v1** | 0.594 | 0.088 |
| **v2** | **0.699** | **0.062** |

---

## v1 — Baseline (2026-05-10)

**Config:** `examples/gan.json`

**Architecture:**
- Generator: `[128, 96, 64]`, noise_dim=12, dropout=0.08, lr=0.0015, batch=64
- Warmup: 20 ep | Rounds: 2 × 16 ep | Early stop: 10
- NEAT: 4 generations, pop=80 | BNN: 60 ep, mc=24
- MC samples (gen): 24 | adv_weight=0.25 | abrams_weight=0.06

**Results (validation, pooled):**
- MAE=2.054 MPa, RMSE=2.577, MAPE=9.74%, R²=0.913
- PICP95=0.531 (calibration poor, model underestimates uncertainty)
- GOST 28d: 0/40 violations

**Weaknesses identified:**
1. MAPE ~10% — no explicit relative-error objective
2. Day-3 and Day-28 R²≈0.44–0.46 — weakest periods
3. PICP95=0.531 (expected 0.95) — BNN too confident
4. Batch=64 → only ~2.5 batches/epoch for N_train=160
5. Only 2 GAN rounds, 20 warmup epochs — undertrained

---

## v2 — Improved (2026-05-10)

**Config:** `examples/gan_v2.json`

**Changes vs v1:**
| Parameter | v1 | v2 | Reason |
|-----------|----|----|--------|
| `hidden_layers` | [128,96,64] | [256,192,128,64] | More capacity |
| `batch_size` | 64 | 32 | 5 batches/epoch vs 2.5 |
| `warmup_epochs` | 20 | 60 | 3× more supervised warmup |
| `epochs_per_round` | 16 | 40 | 2.5× more per GAN round |
| `early_stopping_rounds` | 10 | 15 | More patience |
| `noise_dim` | 12 | 16 | Richer stochasticity |
| `dropout` | 0.08 | 0.05 | Less regularization (small N) |
| `learning_rate` | 0.0015 | 0.001 | Slower, more stable |
| `supervised_weight` | 1.0 | 1.2 | Prioritise supervised signal |
| `adversarial_weight` | 0.25 | 0.15 | Reduce adversarial pressure |
| `mape_weight` | 0.0 | 0.20 | **NEW** — explicit MAPE loss |
| `abrams_weight` | 0.06 | 0.10 | Stronger physics |
| `rounds` | 2 | 3 | More GAN rounds |
| `generator_mc_samples` | 24 | 64 | Better MC estimate |
| `limit_generations` | 4 | 8 | More NEAT search |
| `bnn_mc_samples` | 24 | 64 | Better uncertainty |
| `bnn_epochs` | 60 | 100 | More BNN training |
| `kl_warmup_steps` | 4 | 8 | Wider initial posterior → better PICP |
| `initial_rho` | -3.0 | -2.5 | Wider initial σ → better calibration |
| `fake_jitter_std` | 0.2 | 0.15 | Tighter fake samples |

**Results (validation, pooled):**
- MAE=1.001 MPa, RMSE=1.287, MAPE=3.95%, R²=0.978
- PICP95=0.625 (улучшилось с 0.531, но ещё далеко от 0.95)
- GOST 28d: 0/40 violations
- Discriminator realness: mean=0.699, std=0.062 (v1: 0.594, 0.088) — сгенерированные смеси стали «реалистичнее»

**Per-period (validation):**
| Срок | MAE, МПа | R² |
|------|----------|-----|
| 1 сут | 0.477 | 0.969 |
| 3 сут | 0.813 | 0.946 |
| 7 сут | 1.172 | 0.867 |
| 28 сут | 1.544 | 0.719 |

**Выводы v2:**
- MAPE снизилась с 9.74% → 3.95% (−59%) — MAPE loss сработал
- MAE снизился на 51%, RMSE на 50%
- R² вырос с 0.913 → 0.978
- Слабое место остаётся Day-28 (R²=0.719) — следует рассмотреть отдельный head с бо́льшим весом
- PICP95=0.625 — неплохо лучше, но модель всё ещё недооценивает неопределённость

---

## v3 — Period Weights + Wider Uncertainty (2026-05-10)

**Config:** `examples/gan_v3.json`

**Changes vs v2:**
| Parameter | v2 | v3 | Reason |
|-----------|----|----|--------|
| `period_weights` | — | [1.0, 1.2, 1.5, 2.5] | **NEW** — extra pressure on Day-28 in SmoothL1 + MAPE |
| `noise_dim` | 16 | 24 | Richer MC diversity |
| `dropout` | 0.05 | 0.10 | More MC uncertainty spread |
| `mape_weight` | 0.20 | 0.25 | Slightly stronger MAPE term |
| `learning_rate` | 0.001 | 0.0008 | Slower optimisation |
| `hidden_layers` | [256,192,128,64] | [256,256,128,64] | Larger first hidden block |
| `warmup_epochs` | 60 | 80 | More supervised warmup |
| `epochs_per_round` | 40 | 50 | More per GAN round |
| `rounds` | 3 | 4 | One more GAN round |
| `bnn_epochs` | 100 | 120 | More BNN training |
| `prior_std` | 1.0 | 0.8 | Wider BNN prior → looser posterior |
| `kl_weight` | 0.02 | 0.01 | Weaker KL → more freedom |
| `kl_warmup_steps` | 8 | 12 | Longer warmup → wider early posterior |
| `initial_rho` | -2.5 | -2.0 | Wider initial σ → better PICP |
| `abrams_weight` | 0.10 | 0.12 | Slightly stronger physics |

**Results (validation, pooled):**
- MAE=1.117 MPa, RMSE=1.449, MAPE=4.90%, R²=0.972
- **PICP95=0.825** (+0.200 vs v2) ← main target, significant progress
- GOST 28d: 0/40 violations
- Discriminator realness: mean=0.719, std=0.108

**Per-period (validation):**
| Срок | MAE, МПа | R² | MAPE |
|------|----------|-----|------|
| 1 сут | 1.096 | 0.808 | 7.42% |
| 3 сут | 0.958 | 0.917 | 4.57% |
| 7 сут | 1.101 | 0.872 | 3.92% |
| **28 сут** | **1.313** | **0.782** | **3.67%** |

**Выводы v3:**
- Day-28 R² улучшился: 0.719 → 0.782 (+8.8%) — `period_weights` сработали
- PICP95 значительно улучшился: 0.625 → 0.825 (+32%) — более широкая апостериорная BNN
- Общие MAE/R² незначительно ухудшились (+12% MAE, −0.006 R²) — нормальный trade-off
- Day-1 ухудшился (R²=0.808 vs 0.969) — cost of upweighting Day-28
- Remaining gap: PICP95=0.825 vs target 0.95 — возможно рассмотреть температурное масштабирование или conformal prediction

---

## v4 — Extended Domain Features (2026-05-10)

**Config:** `examples/gan_v4.json`

**Changes vs v3:** `add_extended_features: true` — добавлены 4 новых признака:
- `log_cw_ratio` = ln(c/w) — прямая форма закона Абрамса
- `effective_wc_ratio` = w/(c + 0.4·fly_ash + microsilica) — эффективный w/c с учётом пуццоланов
- `aggregate_binder_ratio` = (sand+gravel)/binder — плотность заполнителей
- `plasticizer_binder_pct` = plasticizer/binder×100 — дозировка пластификатора

Итого: **14 признаков** (было 10). Все гиперпараметры идентичны v3.

**Results (validation, pooled):**
- MAE=1.054 MPa, RMSE=1.408, MAPE=4.90%, R²=0.974
- PICP95=0.725 (хуже v3; дополнительные признаки сжали дисперсию модели)
- GOST 28d: 0/40 violations
- Discriminator realness: mean=0.414 (↓ vs v3=0.719 — генератор стал менее убедительным на расширенном пространстве признаков)
- Feature names: cement, sand, gravel, water, plasticizer_kg, fly_ash, microsilica_kg, water_cement_ratio, cement_water_ratio, log_cw_ratio, effective_wc_ratio, aggregate_binder_ratio, plasticizer_binder_pct, total_mass

**Per-period (validation):**
| Срок | MAE, МПа | R² | MAPE |
|------|----------|-----|------|
| **1 сут** | **0.787** | **0.927** | 6.96% |
| 3 сут | 0.896 | 0.919 | 4.37% |
| 7 сут | 1.231 | 0.810 | 4.58% |
| 28 сут | 1.302 | 0.769 | 3.67% |

**Выводы v4:**
- Day-1 значительно улучшился: R² 0.808 → 0.927 — физические признаки помогли ранней прочности
- Day-28 MAE улучшился: 1.313 → 1.302, R² незначительно упал: 0.782 → 0.769
- Day-7 ухудшился: R² 0.872 → 0.810 — регрессия среднесрочного периода
- PICP95 упал: 0.825 → 0.725 — больше признаков → более уверенные предсказания (меньше spread)
- Лучшая общая точность: R²=0.974 vs v3=0.972, MAE=1.054 vs v3=1.117
- **Вывод**: новые признаки полезны для точности (особенно Day-1), но вредят калибровке (PICP95)

---

## v5 — Config-Only Tuning from v4 (2026-05-10)

**Config:** `examples/gan_v5.json`

**Изменения vs v4:** без изменений кода, только гиперпараметры (более длинное обучение, больше MC сэмплов, усиленный вес Day-28).

**Results (validation, pooled):**
- MAE=1.216 MPa, RMSE=1.546, MAPE=5.53%, R²=0.969
- PICP95=0.750
- Conformal calibration: N/A (модель обучена до добавления conformal кода)
- GOST 28d: 0/40 violations

**Per-period (validation):**
| Срок | MAE, МПа | R² |
|------|----------|-----|
| 1 сут | 0.980 | 0.865 |
| 3 сут | 1.327 | 0.853 |
| 7 сут | 1.272 | 0.833 |
| 28 сут | 1.285 | 0.771 |

**Выводы v5:**
- Переусиление веса Day-28 не дало ожидаемого выигрыша по Day-28 R²
- Общая точность ухудшилась относительно v4 (MAE/RMSE/MAPE)
- PICP95 немного вырос (0.725 → 0.750), но без качественного скачка

---

## v6 — Heteroscedastic Uncertainty + Conformal (2026-05-10)

**Config:** `examples/gan_v6.json`

**Ключевые изменения vs v5:**
- Включён uncertainty head в генераторе (`use_uncertainty_head: true`)
- Добавлен NLL-штраф (`nll_weight: 0.3`) для обучения aleatoric неопределенности
- Активирована conformal калибровка интервалов

**Results (validation, pooled):**
- MAE=0.709 MPa, RMSE=0.931, MAPE=3.10%, R²=0.989
- PICP95=0.988
- **Conformal**: q_hat=1.137, PICP95_conformal=0.944
- Mean uncertainty=1.704 MPa
- GOST 28d: 0/40 violations

**Per-period (validation):**
| Срок | MAE, МПа | R² |
|------|----------|-----|
| 1 сут | 0.708 | 0.941 |
| 3 сут | 0.477 | 0.982 |
| 7 сут | 0.708 | 0.936 |
| 28 сут | 0.945 | 0.856 |

**Выводы v6:**
- Резкий рост качества по всем основным метрикам
- Day-28 R² достиг 0.856 (выше целевого 0.82)
- Покрытие интервалов стало сильно консервативным (PICP95=0.988), а conformal-калибровка вернула покрытие ближе к целевому уровню 0.95

---

## v7 — Larger Model + Stronger Uncertainty (2026-05-10)

**Config:** `examples/gan_v7.json`

**Ключевые изменения vs v6:**
- Более крупная архитектура генератора (`[320,256,192,128,64]`)
- Больше noise-латентов (`noise_dim: 48`)
- Сильнее штраф на uncertainty head (`nll_weight: 0.4`)
- Более агрессивный фокус на поздний срок (`period_weights: [1.0, 1.0, 3.0, 8.0]`)
- Больше MC-сэмплов генератора (`generator_mc_samples: 200`) и ещё один GAN round

**Results (validation, pooled):**
- MAE=0.722 MPa, RMSE=1.150, MAPE=3.72%, R²=0.984
- PICP95=0.956
- **Conformal**: q_hat=1.305, PICP95_conformal=0.956
- Mean uncertainty=1.295 MPa
- GOST 28d: 0/40 violations

**Per-period (validation):**
| Срок | MAE, МПа | R² |
|------|----------|-----|
| 1 сут | 0.833 | 0.871 |
| 3 сут | 0.640 | 0.928 |
| 7 сут | 0.774 | 0.909 |
| 28 сут | 0.638 | 0.911 |

**Выводы v7:**
- Лучшая модель по 28-суточной прочности: Day-28 R²=0.911 и Day-28 MAE=0.638
- Покрытие интервалов почти идеально попало в цель: PICP95=0.956
- Цена за это улучшение: небольшой откат по pooled MAE/RMSE/R² относительно v6
- Если главный приоритет — общая точность, лучше v6; если приоритет — 28 суток и хорошо откалиброванные интервалы, лучше v7

---


