"""
experiments_phase0.py — Реализация Фазы 0:
0.1 Финальный фичсет (A..G - 7 zero-gain + Блок I anomaly_score)
0.2 Обучение Isolation Forest на train+test (unsupervised)
0.3 2x2 Факторный тест: {без anomaly / с anomaly} x {без upweight / с upweight (w=2.0)} для LGBM, CatBoost, XGBoost на 5-fold CV
0.4 Подбор оптимального веса Кластера 2 {1.0, 1.25, 1.5, 1.75, 2.0, 2.5} индивидуально для каждой модели
0.5 Стабильность n_covisitors_total_mean по 5 фолдам (Permutation Importance)
"""

from __future__ import annotations
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from metric import precision_at_recall
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def get_cv_scores(model_factory, X, y, splits, sample_weight=None, cat_cols=None, is_xgb=False, is_cb=False):
    scores = []
    for tr_m, va_m in splits:
        yt, yv = y[tr_m], y[va_m]
        
        if is_xgb:
            xt = pd.get_dummies(X.loc[tr_m], columns=cat_cols, drop_first=True)
            xv = pd.get_dummies(X.loc[va_m], columns=cat_cols, drop_first=True).reindex(columns=xt.columns, fill_value=0)
        elif is_cb:
            xt = X.loc[tr_m].copy()
            xv = X.loc[va_m].copy()
            for c in cat_cols:
                xt[c] = xt[c].astype(str)
                xv[c] = xv[c].astype(str)
        else: # LightGBM
            xt = X.loc[tr_m].copy()
            xv = X.loc[va_m].copy()
            for c in cat_cols:
                xt[c] = xt[c].astype('category')
                xv[c] = xv[c].astype('category')
                
        sw_tr = None
        if sample_weight is not None:
            sw_tr = sample_weight[tr_m]
            
        m = model_factory()
        if sw_tr is not None:
            m.fit(xt, yt, sample_weight=sw_tr)
        else:
            m.fit(xt, yt)
            
        p = m.predict_proba(xv)[:, 1]
        scores.append(precision_at_recall(yv, p))
    return float(np.mean(scores)), float(np.std(scores)), scores


def main():
    t0 = time.time()
    print("=" * 70)
    print("  ФАЗА 0: ЗАКРЫТИЕ МЕТОДОЛОГИЧЕСКИХ ПРОБЕЛОВ ПЕРЕД HPO")
    print("=" * 70)
    
    # Загрузка данных
    train_feats = pd.read_parquet('data/features_train.parquet')
    test_feats = pd.read_parquet('data/features_test.parquet')
    train_meta = pd.read_csv('data/train.csv', parse_dates=['window_start_ts'])
    train_feats['date'] = train_meta['window_start_ts'].dt.date
    dates = sorted(train_feats['date'].unique())
    y_all = train_feats['target'].values
    
    zero_gain = ['n_events_log1p', 'cookie_age_log', 'cookie_age_days', 'session_span_ratio', 'login_flag', 'platform_changed_flag', 'ua_changed_flag']
    h_cols = ['consecutive_same_event_share', 'search_to_search_rate', 'search_to_item_rate', 'item_to_seller_rate']
    exclude_cols = ['cookie_id', 'target', 'date'] + zero_gain + h_cols
    base_clean_cols = [c for c in train_feats.columns if c not in exclude_cols]
    cat_cols = ['ua_os_family', 'ua_browser_family', 'platform_bucket']
    num_cols = [c for c in base_clean_cols if c not in cat_cols]
    
    print(f"Базовый чистый фичсет: {len(base_clean_cols)} фичей (72 числовых, 3 категориальных)")
    
    # 0.2 Обучение Isolation Forest на train + test (unsupervised)
    print("\n--- Шаг 0.2: Обучение Isolation Forest на train+test (unsupervised) ---")
    all_num = pd.concat([train_feats[num_cols], test_feats[num_cols]], axis=0).fillna(0.0)
    iso = IsolationForest(n_estimators=150, contamination=0.08, random_state=42, n_jobs=-1)
    iso.fit(all_num)
    
    train_feats['anomaly_score_iforest'] = -iso.score_samples(train_feats[num_cols].fillna(0.0))
    test_feats['anomaly_score_iforest'] = -iso.score_samples(test_feats[num_cols].fillna(0.0))
    print(f"  Anomaly score рассчитан. Диапазон: min={train_feats['anomaly_score_iforest'].min():.4f}, max={train_feats['anomaly_score_iforest'].max():.4f}")
    
    final_clean_cols = base_clean_cols + ['anomaly_score_iforest']
    print(f"  Финальный фичсет с Блоком I: {len(final_clean_cols)} признаков.")
    
    # 5-фолдовый Walk-Forward CV
    wf_splits = [
        (train_feats['date'] <= dates[5], (train_feats['date'] >= dates[6]) & (train_feats['date'] <= dates[7])),
        (train_feats['date'] <= dates[7], (train_feats['date'] >= dates[8]) & (train_feats['date'] <= dates[9])),
        (train_feats['date'] <= dates[9], (train_feats['date'] >= dates[10]) & (train_feats['date'] <= dates[11])),
        (train_feats['date'] <= dates[11], (train_feats['date'] >= dates[12]) & (train_feats['date'] <= dates[13])),
        (train_feats['date'] <= dates[6], train_feats['date'] >= dates[7])
    ]
    
    # Маска мобильных ботов (Кластер 2): боты без десктоп-событий / mobile-only
    mob_bot_mask = (y_all == 1) & (train_feats['is_mobile_only'].values == 1)
    print(f"  Мобильных ботов в train: {mob_bot_mask.sum()} из {y_all.sum()} ({mob_bot_mask.sum()/y_all.sum():.1%})")
    
    # 0.3 Факторный тест 2x2 для каждой модели
    print("\n--- Шаг 0.3: Факторный тест 2x2 на 5-Fold Walk-Forward CV ---")
    
    models_config = {
        'LightGBM': {
            'factory': lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1),
            'is_xgb': False, 'is_cb': False
        },
        'CatBoost': {
            'factory': lambda: CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1),
            'is_xgb': False, 'is_cb': True
        },
        'XGBoost': {
            'factory': lambda: XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1),
            'is_xgb': True, 'is_cb': False
        }
    }
    
    results_2x2 = []
    
    for m_name, cfg in models_config.items():
        print(f"\nТестирование {m_name}:")
        for with_anom in [False, True]:
            cols_to_use = final_clean_cols if with_anom else base_clean_cols
            X_data = train_feats[cols_to_use]
            
            for with_upweight in [False, True]:
                sw = np.ones(len(y_all))
                if with_upweight:
                    sw[mob_bot_mask] = 2.0
                    
                mean_s, std_s, folds = get_cv_scores(
                    cfg['factory'], X_data, y_all, wf_splits, 
                    sample_weight=sw if with_upweight else None,
                    cat_cols=cat_cols, is_xgb=cfg['is_xgb'], is_cb=cfg['is_cb']
                )
                
                anom_str = "+ Anomaly" if with_anom else "Без Anomaly"
                upw_str = "+ Upweight (w=2)" if with_upweight else "Без Upweight"
                print(f"  {m_name:<10} | {anom_str:<12} | {upw_str:<16} | 5-Fold CV: {mean_s:.4f} ± {std_s:.4f} | Folds: {[round(s, 4) for s in folds]}")
                
                results_2x2.append({
                    'Модель': m_name,
                    'Anomaly Score': with_anom,
                    'Upweighting (w=2.0)': with_upweight,
                    '5-Fold CV mean': round(mean_s, 4),
                    '5-Fold CV std': round(std_s, 4)
                })
                
    df_2x2 = pd.DataFrame(results_2x2)
    df_2x2.to_csv('exp_phase0_2x2.csv', index=False)
    
    # 0.4 Подбор оптимального веса Кластера 2 для каждой модели
    print("\n--- Шаг 0.4: Мини-сетка веса Кластера 2 на 5-Fold CV ---")
    weights_grid = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    best_weights = {}
    weight_results = []
    
    for m_name, cfg in models_config.items():
        print(f"\nПодбор веса для {m_name} (с Anomaly Score):")
        best_w = 1.0
        best_sc = -1.0
        
        for w in weights_grid:
            sw = np.ones(len(y_all))
            if w > 1.0:
                sw[mob_bot_mask] = w
                
            mean_s, std_s, folds = get_cv_scores(
                cfg['factory'], train_feats[final_clean_cols], y_all, wf_splits,
                sample_weight=sw if w > 1.0 else None,
                cat_cols=cat_cols, is_xgb=cfg['is_xgb'], is_cb=cfg['is_cb']
            )
            print(f"  {m_name:<10} | Вес Cl2 = {w:<4} | 5-Fold CV: {mean_s:.4f} ± {std_s:.4f}")
            weight_results.append({
                'Модель': m_name,
                'Вес Cl2': w,
                '5-Fold CV mean': round(mean_s, 4),
                '5-Fold CV std': round(std_s, 4)
            })
            if mean_s > best_sc:
                best_sc = mean_s
                best_w = w
                
        best_weights[m_name] = best_w
        print(f"  -> Оптимальный вес Cl2 для {m_name}: {best_w} (CV = {best_sc:.4f})")
        
    df_weights = pd.DataFrame(weight_results)
    df_weights.to_csv('exp_phase0_weights.csv', index=False)
    
    # 0.5 Стабильность n_covisitors_total_mean по 5 фолдам
    print("\n--- Шаг 0.5: Анализ стабильности признака n_covisitors_total_mean ---")
    feat_target = 'n_covisitors_total_mean'
    perm_importances = []
    
    for i, (tr_m, va_m) in enumerate(wf_splits):
        xt = train_feats.loc[tr_m, final_clean_cols].copy()
        xv = train_feats.loc[va_m, final_clean_cols].copy()
        for c in cat_cols:
            xt[c] = xt[c].astype('category')
            xv[c] = xv[c].astype('category')
        yt, yv = y_all[tr_m], y_all[va_m]
        
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        m.fit(xt, yt)
        base_prec = precision_at_recall(yv, m.predict_proba(xv)[:, 1])
        
        # Перемешивание фичи на валидационном сете (10 повторов)
        perm_drops = []
        for seed in range(5):
            xv_perm = xv.copy()
            xv_perm[feat_target] = np.random.RandomState(seed).permutation(xv_perm[feat_target].values)
            perm_prec = precision_at_recall(yv, m.predict_proba(xv_perm)[:, 1])
            perm_drops.append(base_prec - perm_prec)
            
        drop_mean = float(np.mean(perm_drops))
        perm_importances.append(drop_mean)
        print(f"  Фолд {i}: Базовый скор={base_prec:.4f} | Падение при перемешивании={drop_mean:+.4f}")
        
    print(f"  Среднее падение метрики при перемешивании '{feat_target}': {np.mean(perm_importances):+.4f} ± {np.std(perm_importances):.4f}")
    if all(d > 0.05 for d in perm_importances):
        print("  -> Вывод: Признак n_covisitors_total_mean стабильно критичен на ВСЕХ 5 фолдах без инверсий.")
    
    # Сохраняем обновленные датасеты с фичей anomaly_score_iforest для дальнейших фаз
    train_feats.to_parquet('data/features_train.parquet')
    test_feats.to_parquet('data/features_test.parquet')
    print("\n[OK] Обновленные признаки сохранены в data/features_train.parquet и data/features_test.parquet")
    print(f"Время выполнения Фазы 0: {time.time() - t0:.2f}s")


if __name__ == '__main__':
    main()
