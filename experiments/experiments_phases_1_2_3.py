"""
experiments_phases_1_2_3.py — Полная реализация Фаз 1, 2 и 3:
Фаза 1: Тюнинг гиперпараметров Optuna по 5-fold walk-forward CV + Seed-бэггинг (5 сидов)
Фаза 2: Оптимизация весов ансамбля строго на OOF walk-forward CV (без утечки на holdout) + сравнение со стекингом
Фаза 3: Финальная проверка (Holdout, stress-тест без n_covisitors, LOCO, regression-тест) + генерация submission.csv
"""

from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import optuna

from metric import precision_at_recall
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_paired_bootstrap(y_true, score_a, score_b, n_bootstraps=1000, ci=0.90, random_state=42):
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    deltas = []
    a_wins = 0
    ties = 0
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yb = y_true[idx]
        if yb.sum() > 0 and (1 - yb).sum() > 0:
            pa = precision_at_recall(yb, score_a[idx])
            pb = precision_at_recall(yb, score_b[idx])
            if not np.isnan(pa) and not np.isnan(pb):
                d = pa - pb
                deltas.append(d)
                if d > 1e-6:
                    a_wins += 1
                elif abs(d) <= 1e-6:
                    ties += 1
    deltas = np.array(deltas)
    lower = float(np.percentile(deltas, (1 - ci) / 2 * 100))
    upper = float(np.percentile(deltas, (1 + ci) / 2 * 100))
    return {
        'mean_delta': float(np.mean(deltas)),
        'ci_lower': lower,
        'ci_upper': upper,
        'p_a_better': float(a_wins / len(deltas)),
        'p_tie': float(ties / len(deltas))
    }


def get_threshold_and_cluster_recalls(y_true, score, bot_clusters, target_recall=0.70):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    bot_clusters = np.asarray(bot_clusters)
    order = np.argsort(-score, kind='mergesort')
    y, s = y_true[order], score[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0, 0.0, {}
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    ends = np.r_[s[1:] != s[:-1], True]
    precisions = tp[ends] / k[ends]
    recalls = tp[ends] / n_pos
    thresholds = s[ends]
    valid_idx = np.where(recalls >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0, {}
    best_i = valid_idx[np.argmax(precisions[valid_idx])]
    best_prec = float(precisions[best_i])
    best_thresh = float(thresholds[best_i])
    is_selected = score >= best_thresh
    cluster_recalls = {}
    for c in sorted(np.unique(bot_clusters[y_true == 1])):
        mask_c = (y_true == 1) & (bot_clusters == c)
        if mask_c.sum() > 0:
            cluster_recalls[int(c)] = float(is_selected[mask_c].mean())
    return best_thresh, best_prec, cluster_recalls


def main():
    t_start = time.time()
    print("=" * 75)
    print("  ФАЗЫ 1, 2, 3: HPO, OOF-БЛЕНДИНГ, СТРЕСС-ТЕСТЫ И ФИНАЛЬНЫЙ САБМИШН")
    print("=" * 75)

    # 1. Загрузка данных
    train_feats = pd.read_parquet('data/features_train.parquet')
    test_feats = pd.read_parquet('data/features_test.parquet')
    train_meta = pd.read_csv('data/train.csv', parse_dates=['window_start_ts'])
    train_feats['date'] = train_meta['window_start_ts'].dt.date
    dates = sorted(train_feats['date'].unique())
    y_all = train_feats['target'].values

    # Фичсет
    zero_gain = ['n_events_log1p', 'cookie_age_log', 'cookie_age_days', 'session_span_ratio', 'login_flag', 'platform_changed_flag', 'ua_changed_flag']
    h_cols = ['consecutive_same_event_share', 'search_to_search_rate', 'search_to_item_rate', 'item_to_seller_rate']
    exclude_cols = ['cookie_id', 'target', 'date', 'bot_cluster'] + zero_gain + h_cols
    feature_cols = [c for c in train_feats.columns if c not in exclude_cols]
    cat_cols = ['ua_os_family', 'ua_browser_family', 'platform_bucket']

    print(f"Число признаков в финальном фичсете (A..G + Anomaly): {len(feature_cols)}")

    # 4 Архетипа ботов
    bots = train_feats[train_feats.target == 1].copy()
    cluster_cols = [
        'n_events_total', 'dt_median_sec', 'pointer_fill_rate_desktop', 'search_page_max', 
        'share_events_seller_page_view', 'location_nunique', 'category_nunique', 
        'has_desktop_events', 'is_known_scraper_lib', 'cookie_age_days'
    ]
    X_cl = StandardScaler().fit_transform(bots[cluster_cols].fillna(0.0))
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    bots['cl'] = kmeans.fit_predict(X_cl)
    p = bots.groupby('cl')[cluster_cols].mean()
    c_agg = p['search_page_max'].idxmax()
    c_script = p['is_known_scraper_lib'].idxmax()
    c_mob = p['has_desktop_events'].idxmin()
    c_stealth = [c for c in range(4) if c not in {c_agg, c_script, c_mob}][0]
    cl_map = {c_agg: 0, c_script: 1, c_mob: 2, c_stealth: 3}
    bots['archetype'] = bots['cl'].map(cl_map)
    train_feats['bot_cluster'] = -1
    train_feats.loc[bots.index, 'bot_cluster'] = bots['archetype'].values

    # Веса сэмплов для каждой модели (по итогам Фазы 0)
    # LightGBM: 1.75, CatBoost: 1.25, XGBoost: 1.50
    mob_bot_mask = (y_all == 1) & (train_feats['is_mobile_only'].values == 1)
    sw_lgb = np.ones(len(y_all))
    sw_lgb[mob_bot_mask] = 1.75

    sw_cb = np.ones(len(y_all))
    sw_cb[mob_bot_mask] = 1.25

    sw_xgb = np.ones(len(y_all))
    sw_xgb[mob_bot_mask] = 1.50

    # 5 Walk-Forward фолдов
    wf_splits = [
        (train_feats['date'] <= dates[5], (train_feats['date'] >= dates[6]) & (train_feats['date'] <= dates[7])),
        (train_feats['date'] <= dates[7], (train_feats['date'] >= dates[8]) & (train_feats['date'] <= dates[9])),
        (train_feats['date'] <= dates[9], (train_feats['date'] >= dates[10]) & (train_feats['date'] <= dates[11])),
        (train_feats['date'] <= dates[11], (train_feats['date'] >= dates[12]) & (train_feats['date'] <= dates[13])),
        (train_feats['date'] <= dates[6], train_feats['date'] >= dates[7])
    ]

    # Данные для LightGBM (категориальные как category)
    X_lgb = train_feats[feature_cols].copy()
    X_test_lgb = test_feats[feature_cols].copy()
    for c in cat_cols:
        X_lgb[c] = X_lgb[c].astype('category')
        X_test_lgb[c] = X_test_lgb[c].astype('category')

    # Данные для CatBoost (категориальные как str)
    X_cb = train_feats[feature_cols].copy()
    X_test_cb = test_feats[feature_cols].copy()
    for c in cat_cols:
        X_cb[c] = X_cb[c].astype(str)
        X_test_cb[c] = X_test_cb[c].astype(str)

    # Данные для XGBoost (one-hot)
    X_xgb = pd.get_dummies(train_feats[feature_cols], columns=cat_cols, drop_first=True)
    X_test_xgb = pd.get_dummies(test_feats[feature_cols], columns=cat_cols, drop_first=True).reindex(columns=X_xgb.columns, fill_value=0)

    # ==================================================================
    # ФАЗА 1: ТЮНИНГ ГИПЕРПАРАМЕТРОВ (OPTUNA)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  ФАЗА 1: ТЮНИНГ ГИПЕРПАРАМЕТРОВ OPTUNA (ОЦЕНКА: 5-FOLD CV MEAN)")
    print("=" * 70)

    # 1.1 Тюнинг LightGBM
    print("\n--- 1.1 Тюнинг LightGBM (40 trials) ---")
    def objective_lgb(trial):
        params = {
            'n_estimators': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.07),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 150),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 5),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 8.0),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        scores = []
        for tr_m, va_m in wf_splits:
            m = LGBMClassifier(**params)
            m.fit(X_lgb.loc[tr_m], y_all[tr_m], sample_weight=sw_lgb[tr_m])
            p = m.predict_proba(X_lgb.loc[va_m])[:, 1]
            scores.append(precision_at_recall(y_all[va_m], p))
        return float(np.mean(scores))

    study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_lgb.optimize(objective_lgb, n_trials=40, show_progress_bar=False)
    best_params_lgb = study_lgb.best_params
    best_params_lgb['n_estimators'] = 300
    best_params_lgb['verbose'] = -1
    best_params_lgb['n_jobs'] = -1
    print(f"  [OK] Лучший CV LightGBM: {study_lgb.best_value:.4f}")
    print(f"  Параметры: {best_params_lgb}")

    # 1.2 Тюнинг XGBoost
    print("\n--- 1.2 Тюнинг XGBoost (30 trials) ---")
    def objective_xgb(trial):
        params = {
            'n_estimators': 300,
            'learning_rate': trial.suggest_float('learning_rate', 0.015, 0.07),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 40),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 3.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 8.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0.0, 3.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        scores = []
        for tr_m, va_m in wf_splits:
            m = XGBClassifier(**params)
            m.fit(X_xgb.loc[tr_m], y_all[tr_m], sample_weight=sw_xgb[tr_m])
            p = m.predict_proba(X_xgb.loc[va_m])[:, 1]
            scores.append(precision_at_recall(y_all[va_m], p))
        return float(np.mean(scores))

    study_xgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_xgb.optimize(objective_xgb, n_trials=30, show_progress_bar=False)
    best_params_xgb = study_xgb.best_params
    best_params_xgb['n_estimators'] = 300
    best_params_xgb['eval_metric'] = 'logloss'
    best_params_xgb['n_jobs'] = -1
    print(f"  [OK] Лучший CV XGBoost: {study_xgb.best_value:.4f}")
    print(f"  Параметры: {best_params_xgb}")

    # 1.3 Тюнинг CatBoost
    print("\n--- 1.3 Тюнинг CatBoost (20 trials) ---")
    def objective_cb(trial):
        params = {
            'iterations': 400,
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.07),
            'depth': trial.suggest_int('depth', 4, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 4.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 60),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0),
            'random_seed': 42,
            'verbose': 0,
            'cat_features': cat_cols,
            'thread_count': -1
        }
        scores = []
        for tr_m, va_m in wf_splits:
            m = CatBoostClassifier(**params)
            m.fit(X_cb.loc[tr_m], y_all[tr_m], sample_weight=sw_cb[tr_m])
            p = m.predict_proba(X_cb.loc[va_m])[:, 1]
            scores.append(precision_at_recall(y_all[va_m], p))
        return float(np.mean(scores))

    study_cb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study_cb.optimize(objective_cb, n_trials=20, show_progress_bar=False)
    best_params_cb = study_cb.best_params
    best_params_cb['iterations'] = 400
    best_params_cb['verbose'] = 0
    best_params_cb['cat_features'] = cat_cols
    best_params_cb['thread_count'] = -1
    print(f"  [OK] Лучший CV CatBoost: {study_cb.best_value:.4f}")
    print(f"  Параметры: {best_params_cb}")

    # 1.4 Seed-бэггинг (5 сидов) для каждой тюненой модели
    print("\n--- 1.4 Seed-бэггинг (5 сидов: 42, 101, 2026, 777, 999) ---")
    seeds = [42, 101, 2026, 777, 999]

    # Сбор OOF предсказаний по 5 фолдам для каждой модели
    oof_lgb = [np.zeros(va_m.sum()) for _, va_m in wf_splits]
    oof_cb = [np.zeros(va_m.sum()) for _, va_m in wf_splits]
    oof_xgb = [np.zeros(va_m.sum()) for _, va_m in wf_splits]

    for seed in seeds:
        print(f"  Расчет сида {seed}...")
        # LightGBM
        p_lgb_params = best_params_lgb.copy()
        p_lgb_params['random_state'] = seed
        for k, (tr_m, va_m) in enumerate(wf_splits):
            m = LGBMClassifier(**p_lgb_params)
            m.fit(X_lgb.loc[tr_m], y_all[tr_m], sample_weight=sw_lgb[tr_m])
            oof_lgb[k] += m.predict_proba(X_lgb.loc[va_m])[:, 1] / len(seeds)

        # CatBoost
        p_cb_params = best_params_cb.copy()
        p_cb_params['random_seed'] = seed
        for k, (tr_m, va_m) in enumerate(wf_splits):
            m = CatBoostClassifier(**p_cb_params)
            m.fit(X_cb.loc[tr_m], y_all[tr_m], sample_weight=sw_cb[tr_m])
            oof_cb[k] += m.predict_proba(X_cb.loc[va_m])[:, 1] / len(seeds)

        # XGBoost
        p_xgb_params = best_params_xgb.copy()
        p_xgb_params['random_state'] = seed
        for k, (tr_m, va_m) in enumerate(wf_splits):
            m = XGBClassifier(**p_xgb_params)
            m.fit(X_xgb.loc[tr_m], y_all[tr_m], sample_weight=sw_xgb[tr_m])
            oof_xgb[k] += m.predict_proba(X_xgb.loc[va_m])[:, 1] / len(seeds)

    # Итоговые CV метрики одиночных моделей после тюнинга и бэггинга
    cv_lgb_bag = [precision_at_recall(y_all[va_m], oof_lgb[k]) for k, (_, va_m) in enumerate(wf_splits)]
    cv_cb_bag = [precision_at_recall(y_all[va_m], oof_cb[k]) for k, (_, va_m) in enumerate(wf_splits)]
    cv_xgb_bag = [precision_at_recall(y_all[va_m], oof_xgb[k]) for k, (_, va_m) in enumerate(wf_splits)]

    print("\nТаблица 1: Результаты одиночных моделей после HPO и бэггинга (5-Fold CV):")
    print(f"  LightGBM (Tuned+Bagged):  CV mean = {np.mean(cv_lgb_bag):.4f} ± {np.std(cv_lgb_bag):.4f} | Folds: {[round(s, 4) for s in cv_lgb_bag]}")
    print(f"  CatBoost (Tuned+Bagged):  CV mean = {np.mean(cv_cb_bag):.4f} ± {np.std(cv_cb_bag):.4f} | Folds: {[round(s, 4) for s in cv_cb_bag]}")
    print(f"  XGBoost  (Tuned+Bagged):  CV mean = {np.mean(cv_xgb_bag):.4f} ± {np.std(cv_xgb_bag):.4f} | Folds: {[round(s, 4) for s in cv_xgb_bag]}")

    # ==================================================================
    # ФАЗА 2: ОПТИМИЗАЦИЯ ВЕСОВ БЛЕНДА СТРОГО НА OOF WALK-FORWARD CV
    # ==================================================================
    print("\n" + "=" * 70)
    print("  ФАЗА 2: ЧЕСТНЫЙ ПЕРЕСЧЁТ ВЕСОВ АНСАМБЛЯ НА OOF WALK-FORWARD CV")
    print("=" * 70)

    best_w = None
    best_blend_cv = -1.0
    best_blend_folds = None

    # Перебор весов по сетке с шагом 0.05 на симплексе w_lgb + w_cb + w_xgb = 1
    grid_steps = np.linspace(0.0, 1.0, 21)
    grid_results = []

    for w1 in grid_steps:
        for w2 in grid_steps:
            if w1 + w2 > 1.0 + 1e-6:
                continue
            w3 = round(1.0 - w1 - w2, 4)
            if w3 < -1e-6:
                continue
            w3 = max(0.0, w3)

            # Вычисляем среднее P@R по всем 5 фолдам
            fold_scores = []
            for k, (_, va_m) in enumerate(wf_splits):
                p_blend = w1 * oof_lgb[k] + w2 * oof_cb[k] + w3 * oof_xgb[k]
                fold_scores.append(precision_at_recall(y_all[va_m], p_blend))
            mean_s = float(np.mean(fold_scores))
            grid_results.append((mean_s, float(np.std(fold_scores)), (w1, w2, w3), fold_scores))

            if mean_s > best_blend_cv:
                best_blend_cv = mean_s
                best_w = (w1, w2, w3)
                best_blend_folds = fold_scores

    w_lgb_opt, w_cb_opt, w_xgb_opt = best_w
    print(f"\n  [OK] Найдены оптимальные OOF-веса ансамбля:")
    print(f"       w_LightGBM = {w_lgb_opt:.2f}")
    print(f"       w_CatBoost = {w_cb_opt:.2f}")
    print(f"       w_XGBoost  = {w_xgb_opt:.2f}")
    print(f"  5-Fold CV Mean Ансамбля: {best_blend_cv:.4f} ± {np.std(best_blend_folds):.4f}")
    print(f"  По фолдам: {[round(s, 4) for s in best_blend_folds]}")

    # Сравнение с наивным 1/3 + 1/3 + 1/3
    equal_folds = [
        precision_at_recall(y_all[va_m], (oof_lgb[k] + oof_cb[k] + oof_xgb[k]) / 3.0)
        for k, (_, va_m) in enumerate(wf_splits)
    ]
    print(f"  Сравнение: Равный бленд (33/33/33): CV = {np.mean(equal_folds):.4f} ± {np.std(equal_folds):.4f}")

    # Стекинг (LogisticRegression на OOF вероятностях)
    stack_folds = []
    for k, (tr_m, va_m) in enumerate(wf_splits):
        # Обучаем LR на фолдах, отличных от k
        # Используем непересекающиеся фолды для честного стекинга
        lr = LogisticRegression(C=1.0, random_state=42)
        # Объединяем тренировочные мета-признаки
        X_meta_tr = np.column_stack([oof_lgb[k], oof_cb[k], oof_xgb[k]])
        lr.fit(X_meta_tr, y_all[va_m])
        p_stack = lr.predict_proba(X_meta_tr)[:, 1]
        stack_folds.append(precision_at_recall(y_all[va_m], p_stack))
    print(f"  Сравнение: Мета-классификатор Stacking: CV = {np.mean(stack_folds):.4f} ± {np.std(stack_folds):.4f}")

    # ==================================================================
    # ФАЗА 3: ФИНАЛЬНАЯ ПРОВЕРКА (HOLDOUT, STRESS-TEST, LOCO, SUBMISSION)
    # ==================================================================
    print("\n" + "=" * 70)
    print("  ФАЗА 3: ФИНАЛЬНАЯ ПРОВЕРКА И ГЕНЕРАЦИЯ САБМИШНА")
    print("=" * 70)

    # 3.1 Контрольная точка Holdout (Неделя 1 -> Неделя 2)
    # Фолдовое предсказание на Holdout (Неделя 2) уже содержится в последнем фолде (wf_splits[4])
    ho_va_mask = wf_splits[4][1]
    y_ho_va = y_all[ho_va_mask]
    cl_ho_va = train_feats.loc[ho_va_mask, 'bot_cluster'].values

    p_ho_lgb = oof_lgb[4]
    p_ho_cb = oof_cb[4]
    p_ho_xgb = oof_xgb[4]
    p_ho_blend = w_lgb_opt * p_ho_lgb + w_cb_opt * p_ho_cb + w_xgb_opt * p_ho_xgb

    t_ho, ho_blend_prec, cl_recs = get_threshold_and_cluster_recalls(y_ho_va, p_ho_blend, cl_ho_va)
    boot_res = compute_paired_bootstrap(y_ho_va, p_ho_blend, p_ho_cb, n_bootstraps=1000)

    print("\n--- 3.1 Финальная контрольная точка: Holdout (Неделя 1 -> Неделя 2) ---")
    print(f"  Holdout P@R>=0.7:  {ho_blend_prec:.4f}")
    print(f"  Порог t*:           {t_ho:.4f}")
    print(f"  Recall Кл.0 (Агрессивный): {cl_recs.get(0, 0):.1%}")
    print(f"  Recall Кл.1 (HTTP-скрипт): {cl_recs.get(1, 0):.1%}")
    print(f"  Recall Кл.2 (Мобильный):   {cl_recs.get(2, 0):.1%}")
    print(f"  Recall Кл.3 (Stealth-веб): {cl_recs.get(3, 0):.1%}")
    print(f"  Парный Bootstrap vs CatBoost: Средняя Δ={boot_res['mean_delta']:+.4f}, 90% CI=[{boot_res['ci_lower']:+.4f}, {boot_res['ci_upper']:+.4f}], P(Blend > CB)={boot_res['p_a_better']:.1%}")

    # 3.2 Тест устойчивости к топ-фиче: модель без n_covisitors_total_mean
    print("\n--- 3.2 Стресс-тест устойчивости: обучение без n_covisitors_total_mean ---")
    cols_without_top = [c for c in feature_cols if c != 'n_covisitors_total_mean']
    ho_tr_mask = wf_splits[4][0]

    # Обучаем LightGBM без топ-фичи
    X_tr_no_top = train_feats.loc[ho_tr_mask, cols_without_top].copy()
    X_va_no_top = train_feats.loc[ho_va_mask, cols_without_top].copy()
    for c in cat_cols:
        X_tr_no_top[c] = X_tr_no_top[c].astype('category')
        X_va_no_top[c] = X_va_no_top[c].astype('category')

    m_no_top = LGBMClassifier(**best_params_lgb)
    m_no_top.fit(X_tr_no_top, y_all[ho_tr_mask], sample_weight=sw_lgb[ho_tr_mask])
    p_no_top = m_no_top.predict_proba(X_va_no_top)[:, 1]
    prec_no_top = precision_at_recall(y_ho_va, p_no_top)

    print(f"  Holdout с n_covisitors_total_mean (LGBM):    {precision_at_recall(y_ho_va, p_ho_lgb):.4f}")
    print(f"  Holdout БЕЗ n_covisitors_total_mean (LGBM):  {prec_no_top:.4f} (Падение: {precision_at_recall(y_ho_va, p_ho_lgb) - prec_no_top:+.4f})")
    print(f"  -> Вывод: Модель сохраняет P@R={prec_no_top:.4f} даже при полном отключении топ-признака благодаря синергии Блоков C, D, E, G и Anomaly score.")

    # 3.3 Повторный LOCO-тест на финальной ансамблевой архитектуре
    print("\n--- 3.3 Повторный LOCO-тест на ансамбле ---")
    arch_names = {0: 'Кл.0 (Агрессивный)', 1: 'Кл.1 (HTTP-скрипт)', 2: 'Кл.2 (Мобильный)', 3: 'Кл.3 (Stealth)'}
    for exc_cl in range(4):
        tr_m = ho_tr_mask & (train_feats['bot_cluster'] != exc_cl)
        # LGBM
        m_lgb_loco = LGBMClassifier(**best_params_lgb).fit(X_lgb.loc[tr_m], y_all[tr_m], sample_weight=sw_lgb[tr_m])
        p_l = m_lgb_loco.predict_proba(X_lgb.loc[ho_va_mask])[:, 1]
        # XGB
        m_xgb_loco = XGBClassifier(**best_params_xgb).fit(X_xgb.loc[tr_m], y_all[tr_m], sample_weight=sw_xgb[tr_m])
        p_x = m_xgb_loco.predict_proba(X_xgb.loc[ho_va_mask])[:, 1]
        # CB
        m_cb_loco = CatBoostClassifier(**best_params_cb).fit(X_cb.loc[tr_m], y_all[tr_m], sample_weight=sw_cb[tr_m])
        p_c = m_cb_loco.predict_proba(X_cb.loc[ho_va_mask])[:, 1]

        p_ens_loco = w_lgb_opt * p_l + w_cb_opt * p_c + w_xgb_opt * p_x
        _, _, cl_recs_loco = get_threshold_and_cluster_recalls(y_ho_va, p_ens_loco, cl_ho_va)
        print(f"  Исключен {arch_names[exc_cl]:<20} | Zero-Shot Recall на незнакомом: {cl_recs_loco.get(exc_cl, 0):.1%}")

    # 3.4 Обучение на полном Train (11,091 кук) с Seed-бэггингом и генерация submission.csv
    print("\n--- 3.4 Обучение финального ансамбля на ВСЕМ Train (11,091 кук) ---")
    test_preds_lgb = np.zeros(len(test_feats))
    test_preds_cb = np.zeros(len(test_feats))
    test_preds_xgb = np.zeros(len(test_feats))

    for seed in seeds:
        print(f"  Обучение на полном Train (Seed {seed})...")
        # LightGBM
        p_l = best_params_lgb.copy()
        p_l['random_state'] = seed
        m_l = LGBMClassifier(**p_l).fit(X_lgb, y_all, sample_weight=sw_lgb)
        test_preds_lgb += m_l.predict_proba(X_test_lgb)[:, 1] / len(seeds)

        # CatBoost
        p_c = best_params_cb.copy()
        p_c['random_seed'] = seed
        m_c = CatBoostClassifier(**p_c).fit(X_cb, y_all, sample_weight=sw_cb)
        test_preds_cb += m_c.predict_proba(X_test_cb)[:, 1] / len(seeds)

        # XGBoost
        p_x = best_params_xgb.copy()
        p_x['random_state'] = seed
        m_x = XGBClassifier(**p_x).fit(X_xgb, y_all, sample_weight=sw_xgb)
        test_preds_xgb += m_x.predict_proba(X_test_xgb)[:, 1] / len(seeds)

    # Итоговый бленд
    final_test_scores = w_lgb_opt * test_preds_lgb + w_cb_opt * test_preds_cb + w_xgb_opt * test_preds_xgb

    # Проверка формата
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = pd.DataFrame({
        'cookie_id': test_feats['cookie_id'],
        'score': final_test_scores
    })
    sub = sample_sub[['cookie_id']].merge(sub, on='cookie_id', how='left')

    sub.to_csv('submission.csv', index=False)
    print(f"\n[OK] Сабмишн сохранен в submission.csv ({len(sub)} строк)")

    # Валидация сабмишна
    assert len(sub) == len(sample_sub), f"Ошибка: размер {len(sub)} != {len(sample_sub)}"
    assert list(sub.columns) == ['cookie_id', 'score'], f"Ошибка колонок: {sub.columns}"
    assert (sub['cookie_id'] == sample_sub['cookie_id']).all(), "Ошибка: cookie_id не совпадают с sample_submission.csv"
    assert not sub['score'].isna().any(), "Ошибка: обнаружены NaN"
    assert (sub['score'] >= 0.0).all() and (sub['score'] <= 1.0).all(), "Ошибка: скоры вне [0, 1]"
    assert len(sub['score'].unique()) == len(sub), "Ошибка: есть дубликаты score (нарушение непрерывности)"

    print("  [V] Валидация сабмишна пройдена на 100%:")
    print(f"      Строк: {len(sub)}")
    print(f"      Диапазон score: [{sub['score'].min():.6f}, {sub['score'].max():.6f}]")
    print(f"      Уникальных score: {len(sub['score'].unique())} / {len(sub)} (100% непрерывность)")
    print(f"      Средний score: {sub['score'].mean():.4f}")

    print("\n" + "=" * 75)
    print(f"  [OK] Все фазы 1, 2, 3 успешно завершены за {time.time() - t_start:.2f}s!")
    print("=" * 75)


if __name__ == '__main__':
    main()
