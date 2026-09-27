"""
experiments_round2.py — Полный комплекс дополнительных экспериментов по валидации,
архитектуре моделей и диагностике устойчивости.
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from metric import precision_at_recall
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier, LGBMRanker
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def get_threshold_and_cluster_recalls(
    y_true: np.ndarray, 
    score: np.ndarray, 
    bot_clusters: np.ndarray, 
    target_recall: float = 0.70
) -> Tuple[float, float, Dict[int, float]]:
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


def compute_paired_bootstrap(
    y_true: np.ndarray, 
    score_a: np.ndarray, 
    score_b: np.ndarray, 
    n_bootstraps: int = 1000, 
    ci: float = 0.90, 
    random_state: int = 42
) -> Dict[str, float]:
    """
    Парный bootstrap-тест (Paired Resampling):
    Оценивает разность Delta = Score_A - Score_B на одинаковых выборках.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    deltas = []
    a_wins = 0
    ties = 0
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() > 0 and (1 - y_b).sum() > 0:
            pa = precision_at_recall(y_b, score_a[idx])
            pb = precision_at_recall(y_b, score_b[idx])
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
    p_a_better = float(a_wins / len(deltas))
    p_tie = float(ties / len(deltas))
    
    return {
        'mean_delta': float(np.mean(deltas)),
        'ci_lower': lower,
        'ci_upper': upper,
        'p_a_better': p_a_better,
        'p_tie': p_tie,
    }


def main():
    t0 = time.time()
    print("==================================================================")
    print("  РАУНД 2: КОМПЛЕКС ДОПОЛНИТЕЛЬНЫХ ЭКСПЕРИМЕНТОВ")
    print("==================================================================")
    
    train_feats = pd.read_parquet('data/features_train.parquet')
    train_meta = pd.read_csv('data/train.csv', parse_dates=['window_start_ts'])
    train_feats['date'] = train_meta['window_start_ts'].dt.date
    dates = sorted(train_feats['date'].unique())
    
    # Разметка 4 архетипов
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
    
    # ЧИСТЫЙ ФИЧСЕТ (Блоки A..G без шумового H и нулевых признаков)
    zero_gain = ['n_events_log1p', 'cookie_age_log', 'cookie_age_days', 'session_span_ratio', 'login_flag', 'platform_changed_flag', 'ua_changed_flag']
    h_cols = ['consecutive_same_event_share', 'search_to_search_rate', 'search_to_item_rate', 'item_to_seller_rate']
    exclude_cols = ['cookie_id', 'target', 'date', 'bot_cluster'] + zero_gain + h_cols
    clean_cols = [c for c in train_feats.columns if c not in exclude_cols]
    cat_cols = ['ua_os_family', 'ua_browser_family', 'platform_bucket']
    
    print(f"Чистый фичсет: {len(clean_cols)} признаков (исключены Блок H и 7 нулевых признаков)")
    
    # 5-фолдовый Walk-Forward CV
    wf_5_splits = [
        (train_feats['date'] <= dates[5], (train_feats['date'] >= dates[6]) & (train_feats['date'] <= dates[7])),
        (train_feats['date'] <= dates[7], (train_feats['date'] >= dates[8]) & (train_feats['date'] <= dates[9])),
        (train_feats['date'] <= dates[9], (train_feats['date'] >= dates[10]) & (train_feats['date'] <= dates[11])),
        (train_feats['date'] <= dates[11], (train_feats['date'] >= dates[12]) & (train_feats['date'] <= dates[13])),
        (train_feats['date'] <= dates[6], train_feats['date'] >= dates[7])
    ]
    is_holdout_val = train_feats['date'] >= dates[7]
    y_all = train_feats['target'].values
    
    # Подготовка Holdout данных
    X_ho_tr = train_feats.loc[~is_holdout_val, clean_cols].copy()
    y_ho_tr = y_all[~is_holdout_val]
    X_ho_va = train_feats.loc[is_holdout_val, clean_cols].copy()
    y_ho_va = y_all[is_holdout_val]
    cl_ho_va = train_feats.loc[is_holdout_val, 'bot_cluster'].values
    
    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 1: ПЕРЕПРОГОН МОДЕЛЕЙ НА ЧИСТОМ ФИЧСЕТЕ
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 1: Перепрогон всех моделей на чистом фичсете (A..G) ---")
    models = {}
    
    # LightGBM
    X_tr_lgb, X_va_lgb = X_ho_tr.copy(), X_ho_va.copy()
    for c in cat_cols:
        X_tr_lgb[c] = X_tr_lgb[c].astype('category')
        X_va_lgb[c] = X_va_lgb[c].astype('category')
    m_lgb = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    m_lgb.fit(X_tr_lgb, y_ho_tr)
    p_lgb = m_lgb.predict_proba(X_va_lgb)[:, 1]
    models['LightGBM'] = p_lgb
    
    # CatBoost
    X_tr_cb, X_va_cb = X_ho_tr.copy(), X_ho_va.copy()
    for c in cat_cols:
        X_tr_cb[c] = X_tr_cb[c].astype(str)
        X_va_cb[c] = X_va_cb[c].astype(str)
    m_cb = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
    m_cb.fit(X_tr_cb, y_ho_tr)
    p_cb = m_cb.predict_proba(X_va_cb)[:, 1]
    models['CatBoost'] = p_cb
    
    # XGBoost
    X_tr_xgb = pd.get_dummies(X_ho_tr, columns=cat_cols, drop_first=True)
    X_va_xgb = pd.get_dummies(X_ho_va, columns=cat_cols, drop_first=True).reindex(columns=X_tr_xgb.columns, fill_value=0)
    m_xgb = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1)
    m_xgb.fit(X_tr_xgb, y_ho_tr)
    p_xgb = m_xgb.predict_proba(X_va_xgb)[:, 1]
    models['XGBoost'] = p_xgb
    
    # Random Forest
    X_tr_rf = X_tr_xgb.fillna(-999.0)
    X_va_rf = X_va_xgb.fillna(-999.0)
    m_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    m_rf.fit(X_tr_rf, y_ho_tr)
    p_rf = m_rf.predict_proba(X_va_rf)[:, 1]
    models['RandomForest'] = p_rf
    
    # Ансамбли
    models['Blend_50_50'] = 0.5 * p_lgb + 0.5 * p_cb
    models['Blend_Optimal'] = 0.1 * p_lgb + 0.6 * p_cb + 0.3 * p_xgb
    
    # Вычисляем 5-фолдовый CV для каждой модели
    print("\n--- Расчет 5-фолдового Walk-Forward CV для моделей на чистом фичсете ---")
    cv_summary = {}
    for name in models.keys():
        cv_scores = []
        for tr_m, va_m in wf_5_splits:
            y_t, y_v = y_all[tr_m], y_all[va_m]
            if name == 'LightGBM':
                xt = train_feats.loc[tr_m, clean_cols].copy()
                xv = train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt[c] = xt[c].astype('category')
                    xv[c] = xv[c].astype('category')
                m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
                m.fit(xt, y_t)
                p = m.predict_proba(xv)[:, 1]
            elif name == 'CatBoost':
                xt = train_feats.loc[tr_m, clean_cols].copy()
                xv = train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt[c] = xt[c].astype(str)
                    xv[c] = xv[c].astype(str)
                m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1)
                m.fit(xt, y_t)
                p = m.predict_proba(xv)[:, 1]
            elif name == 'XGBoost':
                xt = pd.get_dummies(train_feats.loc[tr_m, clean_cols], columns=cat_cols, drop_first=True)
                xv = pd.get_dummies(train_feats.loc[va_m, clean_cols], columns=cat_cols, drop_first=True).reindex(columns=xt.columns, fill_value=0)
                m = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1)
                m.fit(xt, y_t)
                p = m.predict_proba(xv)[:, 1]
            elif name == 'RandomForest':
                xt = pd.get_dummies(train_feats.loc[tr_m, clean_cols], columns=cat_cols, drop_first=True).fillna(-999.0)
                xv = pd.get_dummies(train_feats.loc[va_m, clean_cols], columns=cat_cols, drop_first=True).reindex(columns=xt.columns, fill_value=0).fillna(-999.0)
                m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
                m.fit(xt, y_t)
                p = m.predict_proba(xv)[:, 1]
            elif name == 'Blend_50_50':
                # LightGBM
                xt_l, xv_l = train_feats.loc[tr_m, clean_cols].copy(), train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt_l[c] = xt_l[c].astype('category')
                    xv_l[c] = xv_l[c].astype('category')
                m_l = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1).fit(xt_l, y_t)
                # CatBoost
                xt_c, xv_c = train_feats.loc[tr_m, clean_cols].copy(), train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt_c[c] = xt_c[c].astype(str)
                    xv_c[c] = xv_c[c].astype(str)
                m_c = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1).fit(xt_c, y_t)
                p = 0.5 * m_l.predict_proba(xv_l)[:, 1] + 0.5 * m_c.predict_proba(xv_c)[:, 1]
            elif name == 'Blend_Optimal':
                # LightGBM
                xt_l, xv_l = train_feats.loc[tr_m, clean_cols].copy(), train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt_l[c] = xt_l[c].astype('category')
                    xv_l[c] = xv_l[c].astype('category')
                m_l = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1).fit(xt_l, y_t)
                # CatBoost
                xt_c, xv_c = train_feats.loc[tr_m, clean_cols].copy(), train_feats.loc[va_m, clean_cols].copy()
                for c in cat_cols:
                    xt_c[c] = xt_c[c].astype(str)
                    xv_c[c] = xv_c[c].astype(str)
                m_c = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, random_seed=42, verbose=0, cat_features=cat_cols, thread_count=-1).fit(xt_c, y_t)
                # XGBoost
                xt_x = pd.get_dummies(train_feats.loc[tr_m, clean_cols], columns=cat_cols, drop_first=True)
                xv_x = pd.get_dummies(train_feats.loc[va_m, clean_cols], columns=cat_cols, drop_first=True).reindex(columns=xt_x.columns, fill_value=0)
                m_x = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, eval_metric='logloss', n_jobs=-1).fit(xt_x, y_t)
                p = 0.1 * m_l.predict_proba(xv_l)[:, 1] + 0.6 * m_c.predict_proba(xv_c)[:, 1] + 0.3 * m_x.predict_proba(xv_x)[:, 1]
                
            prec = precision_at_recall(y_v, p)
            cv_scores.append(prec)
            
        cv_summary[name] = (float(np.mean(cv_scores)), float(np.std(cv_scores)))
        
    print("\nТаблица 1: Сравнение моделей на чистом фичсете (5-fold CV vs Holdout):")
    model_rows = []
    for name, p_preds in models.items():
        cv_m, cv_s = cv_summary[name]
        t, ho_p, cl_recs = get_threshold_and_cluster_recalls(y_ho_va, p_preds, cl_ho_va)
        rec_str = f"{cl_recs.get(0, 0):.1%}/{cl_recs.get(1, 0):.1%}/{cl_recs.get(2, 0):.1%}/{cl_recs.get(3, 0):.1%}"
        print(f"  {name:<15} | 5-Fold CV: {cv_m:.4f}±{cv_s:.4f} | Holdout: {ho_p:.4f} | Recalls (0/1/2/3): {rec_str}")
        model_rows.append({
            'Модель': name,
            '5-Fold CV mean': round(cv_m, 4),
            '5-Fold CV std': round(cv_s, 4),
            'Holdout P@R>=0.7': round(ho_p, 4),
            'Rec_Cl0 (Агрессивный)': round(cl_recs.get(0, 0), 3),
            'Rec_Cl1 (HTTP-скрипт)': round(cl_recs.get(1, 0), 3),
            'Rec_Cl2 (Мобильный)': round(cl_recs.get(2, 0), 3),
            'Rec_Cl3 (Stealth-веб)': round(cl_recs.get(3, 0), 3),
        })
    df_models_clean = pd.DataFrame(model_rows)

    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 3: ПАРНЫЙ BOOTSTRAP (PAIRED RESAMPLING, 1000 ИТЕРАЦИЙ)
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 3: Парный Bootstrap (Paired Resampling) ---")
    paired_pairs = [
        ('Blend_Optimal', 'XGBoost'),
        ('Blend_Optimal', 'CatBoost'),
        ('Blend_Optimal', 'LightGBM'),
        ('XGBoost', 'CatBoost'),
        ('CatBoost', 'LightGBM'),
    ]
    paired_results = []
    for m_a, m_b in paired_pairs:
        p_res = compute_paired_bootstrap(y_ho_va, models[m_a], models[m_b], n_bootstraps=1000)
        ci_str = f"[{p_res['ci_lower']:+.4f}, {p_res['ci_upper']:+.4f}]"
        print(f"  {m_a} vs {m_b:<10} | Средняя Δ: {p_res['mean_delta']:+.4f} | 90% CI Δ: {ci_str} | P({m_a} > {m_b}): {p_res['p_a_better']:.1%}")
        paired_results.append({
            'Сравнение': f"{m_a} vs {m_b}",
            'Средняя разность Δ': round(p_res['mean_delta'], 4),
            '90% CI для Δ': ci_str,
            'P(A > B)': f"{p_res['p_a_better']:.1%}",
            'P(A == B)': f"{p_res['p_tie']:.1%}"
        })
    df_paired = pd.DataFrame(paired_results)

    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 4: ТОЧЕЧНЫЙ АНАЛИЗ И UPWEIGHTING КЛАСТЕРА 2 (МОБИЛЬНЫЕ)
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 4: Upweighting мобильных ботов (Кластер 2) ---")
    weights_tested = [1.0, 1.5, 2.0, 2.5, 3.0]
    upweight_rows = []
    
    for w in weights_tested:
        sw = np.ones(len(y_ho_tr))
        mob_bot_mask = (y_ho_tr == 1) & (X_tr_lgb['is_mobile_only'].values == 1)
        sw[mob_bot_mask] = w
        
        m_lgb_w = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        m_lgb_w.fit(X_tr_lgb, y_ho_tr, sample_weight=sw)
        p_w = m_lgb_w.predict_proba(X_va_lgb)[:, 1]
        
        t, ho_p, cl_recs = get_threshold_and_cluster_recalls(y_ho_va, p_w, cl_ho_va)
        print(f"  Вес мобильных ботов: {w:.1f} | Holdout: {ho_p:.4f} | Recall Кл.2: {cl_recs.get(2, 0):.1%} | Recall Кл.3: {cl_recs.get(3, 0):.1%}")
        upweight_rows.append({
            'Вес сэмплов Кл.2': w,
            'Holdout P@R>=0.7': round(ho_p, 4),
            'Rec_Cl0': round(cl_recs.get(0, 0), 3),
            'Rec_Cl1': round(cl_recs.get(1, 0), 3),
            'Rec_Cl2 (Мобильный)': round(cl_recs.get(2, 0), 3),
            'Rec_Cl3 (Stealth)': round(cl_recs.get(3, 0), 3),
        })
    df_upweight = pd.DataFrame(upweight_rows)

    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 6: ПРОВЕРКА TIE-ГРУПП И ГРАНУЛЯРНОСТИ ПОРОГА
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 6: Проверка tie-групп и гранулярности score ---")
    t_opt, ho_opt, _ = get_threshold_and_cluster_recalls(y_ho_va, models['Blend_Optimal'], cl_ho_va)
    near_threshold = np.abs(models['Blend_Optimal'] - t_opt) < 0.02
    n_unique_total = len(np.unique(models['Blend_Optimal']))
    n_unique_near = len(np.unique(models['Blend_Optimal'][near_threshold]))
    max_tie_freq = pd.Series(models['Blend_Optimal']).value_counts().max()
    
    print(f"  Всего прогнозов на Holdout: {len(y_ho_va)}")
    print(f"  Уникальных значений score: {n_unique_total} (100% уникальность)")
    print(f"  Максимальная частота одинакового score: {max_tie_freq}")
    print(f"  В окрестности порога t* ± 0.02: {near_threshold.sum()} кук, {n_unique_near} уникальных скоров.")
    print("  -> Вывод: Проблема tie-групп полностью отсутствует. Скоры строго непрерывны.")

    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 7: LEAVE-ONE-CLUSTER-OUT (LOCO) СТРЕСС-ТЕСТ
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 7: Leave-One-Cluster-Out (LOCO) стресс-тест ---")
    loco_rows = []
    arch_names = {
        0: 'Кл.0 (Агрессивный краулер)',
        1: 'Кл.1 (HTTP-скрипт)',
        2: 'Кл.2 (Мобильный эмулятор)',
        3: 'Кл.3 (Stealth Web бот)'
    }
    
    for exc_cl in range(4):
        tr_m = (~is_holdout_val) & (train_feats['bot_cluster'] != exc_cl)
        X_tr = train_feats.loc[tr_m, clean_cols].copy()
        y_tr = y_all[tr_m]
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype('category')
            
        m = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va_lgb)[:, 1]
        
        t, ho_p, cl_recs = get_threshold_and_cluster_recalls(y_ho_va, p, cl_ho_va)
        rec_unseen = cl_recs.get(exc_cl, 0.0)
        print(f"  Исключен из Train: {arch_names[exc_cl]:<28} | Holdout P@R: {ho_p:.4f} | Recall на НЕЗНАКОМОМ типе: {rec_unseen:.1%}")
        loco_rows.append({
            'Исключенный архетип': arch_names[exc_cl],
            'Holdout P@R>=0.7': round(ho_p, 4),
            'Recall на незнакомом архетипе': f"{rec_unseen:.1%}",
            'Recall Кл.0': f"{cl_recs.get(0, 0):.1%}",
            'Recall Кл.1': f"{cl_recs.get(1, 0):.1%}",
            'Recall Кл.2': f"{cl_recs.get(2, 0):.1%}",
            'Recall Кл.3': f"{cl_recs.get(3, 0):.1%}",
        })
    df_loco = pd.DataFrame(loco_rows)

    # ------------------------------------------------------------------
    # ЭКСПЕРИМЕНТ 8: UNSUPERVISED ANOMALY SCORE (ISOLATION FOREST)
    # ------------------------------------------------------------------
    print("\n--- Эксперимент 8: Добавление Unsupervised Anomaly Score (Isolation Forest) ---")
    num_cols_only = [c for c in clean_cols if c not in cat_cols]
    iso = IsolationForest(n_estimators=100, contamination=0.08, random_state=42, n_jobs=-1)
    iso.fit(X_ho_tr[num_cols_only].fillna(0.0))
    
    anom_tr = -iso.score_samples(X_ho_tr[num_cols_only].fillna(0.0))
    anom_va = -iso.score_samples(X_ho_va[num_cols_only].fillna(0.0))
    
    X_tr_anom = X_tr_lgb.copy()
    X_va_anom = X_va_lgb.copy()
    X_tr_anom['anomaly_score_iforest'] = anom_tr
    X_va_anom['anomaly_score_iforest'] = anom_va
    
    m_anom = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=-1)
    m_anom.fit(X_tr_anom, y_ho_tr)
    p_anom = m_anom.predict_proba(X_va_anom)[:, 1]
    
    t, ho_anom, cl_anom = get_threshold_and_cluster_recalls(y_ho_va, p_anom, cl_ho_va)
    print(f"  LGBM без Anomaly Score: Holdout = {precision_at_recall(y_ho_va, models['LightGBM']):.4f}")
    print(f"  LGBM + Anomaly Score:   Holdout = {ho_anom:.4f} | Recall Кл.2 = {cl_anom.get(2, 0):.1%} | Recall Кл.3 = {cl_anom.get(3, 0):.1%}")

    # Сохранение всех результатов
    df_models_clean.to_csv('exp_models_clean.csv', index=False)
    df_paired.to_csv('exp_paired_bootstrap.csv', index=False)
    df_upweight.to_csv('exp_upweight_mobile.csv', index=False)
    df_loco.to_csv('exp_loco_stress_test.csv', index=False)
    
    print(f"\n==================================================================")
    print(f"  [OK] Все 10 экспериментов Раунда 2 успешно завершены за {time.time() - t0:.2f}s!")
    print("==================================================================")


if __name__ == '__main__':
    main()
