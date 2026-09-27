"""
model_selection.py — Комплексный скрипт валидации, инкрементального тестирования блоков фичей
и сравнения семейств моделей.

Протокол валидации:
1. Walk-Forward CV (3 временных фолда внутри train) для оценки стабильности (CV mean ± std).
2. Primary Holdout: Неделя 1 (06.04-12.04) -> Неделя 2 (13.04-19.04).
3. Bootstrap 90% Confidence Interval (1000 итераций ресэмплинга).
4. Диагностический recall по 4 архетипам ботов (Кластер 0, 1, 2, 3).
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

from metric import precision_at_recall
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# Гарантируем UTF-8 вывод на Windows
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
    """
    Вычисляет максимальный precision при recall >= 0.70 и замеряет recall
    отдельно по каждому из 4 архетипов ботов при найденном оптимальном пороге.
    """
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
    
    # Recall каждого кластера при оптимальном пороге
    is_selected = score >= best_thresh
    cluster_recalls = {}
    for c in sorted(np.unique(bot_clusters[y_true == 1])):
        mask_c = (y_true == 1) & (bot_clusters == c)
        if mask_c.sum() > 0:
            cluster_recalls[int(c)] = float(is_selected[mask_c].mean())
            
    return best_thresh, best_prec, cluster_recalls


def compute_bootstrap_ci(
    y_true: np.ndarray, 
    score: np.ndarray, 
    n_bootstraps: int = 1000, 
    ci: float = 0.90, 
    random_state: int = 42
) -> Tuple[float, float]:
    """90% доверительный интервал для P@R>=0.70 через 1000 bootstrap-итераций."""
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    scores = []
    
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() > 0 and (1 - y_b).sum() > 0:
            p = precision_at_recall(y_b, score[idx])
            if not np.isnan(p):
                scores.append(p)
                
    if len(scores) == 0:
        return 0.0, 0.0
        
    lower = float(np.percentile(scores, (1 - ci) / 2 * 100))
    upper = float(np.percentile(scores, (1 + ci) / 2 * 100))
    return lower, upper


def prepare_data_and_clusters() -> Tuple[pd.DataFrame, List[Tuple[np.ndarray, np.ndarray]], np.ndarray, Dict[str, List[str]]]:
    """Загрузка фичей, разметка 4 кластеров ботов и формирование фолдов."""
    print("--- Загрузка признаков и метаданных ---")
    train_feats = pd.read_parquet('data/features_train.parquet')
    train_meta = pd.read_csv('data/train.csv', parse_dates=['window_start_ts'])
    train_feats['date'] = train_meta['window_start_ts'].dt.date
    
    # Разметка 4 архетипов на позитивном классе
    bots = train_feats[train_feats.target == 1].copy()
    cluster_cols = [
        'n_events_total', 'dt_median_sec', 'pointer_fill_rate_desktop', 'search_page_max', 
        'share_events_seller_page_view', 'location_nunique', 'category_nunique', 
        'has_desktop_events', 'is_known_scraper_lib', 'cookie_age_days'
    ]
    X_cl = StandardScaler().fit_transform(bots[cluster_cols].fillna(0.0))
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    bot_labels = kmeans.fit_predict(X_cl)
    
    # Сопоставляем кластеры с архетипами по характерным признакам
    bots['cluster_raw'] = bot_labels
    profiles = bots.groupby('cluster_raw')[cluster_cols].mean()
    
    # 0: Aggressive (высокий page_max, низкий dt)
    c_agg = profiles['search_page_max'].idxmax()
    # 1: Script (is_known_scraper_lib == 1)
    c_script = profiles['is_known_scraper_lib'].idxmax()
    # 2: Mobile (has_desktop_events == 0)
    c_mobile = profiles['has_desktop_events'].idxmin()
    # 3: Stealth (оставшийся с мышью)
    assigned = {c_agg, c_script, c_mobile}
    remaining = [c for c in range(4) if c not in assigned]
    c_stealth = remaining[0] if len(remaining) > 0 else 3
    
    cluster_map = {c_agg: 0, c_script: 1, c_mobile: 2, c_stealth: 3}
    bots['archetype'] = bots['cluster_raw'].map(cluster_map)
    
    train_feats['bot_cluster'] = -1
    train_feats.loc[bots.index, 'bot_cluster'] = bots['archetype'].values
    
    print("Распределение архетипов в train ботах:")
    arch_names = {
        0: 'Кл.0 (Агрессивный Scraper)',
        1: 'Кл.1 (HTTP-скрипт)',
        2: 'Кл.2 (Мобильный эмулятор)',
        3: 'Кл.3 (Stealth Web бот)'
    }
    for a_id in sorted(arch_names.keys()):
        cnt = (train_feats['bot_cluster'] == a_id).sum()
        print(f"  {arch_names[a_id]}: {cnt} ({cnt/len(bots):.1%})")
        
    # Блоки признаков
    blocks = {
        'A': [c for c in train_feats.columns if c.startswith('n_events') or c.startswith('share_events') or c.startswith('n_unique') or c in ['item_view_to_search_ratio', 'login_flag']],
        'B': [c for c in train_feats.columns if c.startswith('cookie_age') or c.startswith('is_fresh')],
        'C': [c for c in train_feats.columns if c.startswith('dt_') or c.startswith('share_dt') or c.startswith('share_instant') or c.startswith('session_') or c.startswith('time_') or c.startswith('night_') or c.startswith('active_hours') or c.startswith('peak_hour') or c.startswith('hourly_') or c == 'events_per_active_hour'],
        'D': [c for c in train_feats.columns if c.startswith('pointer_') or c.startswith('desktop_events') or c == 'has_desktop_events'],
        'E': [c for c in train_feats.columns if c.startswith('ua_') or c.startswith('platform_') or c.startswith('is_known_scraper') or c == 'is_headless_ua'],
        'F': [c for c in train_feats.columns if c.startswith('search_page_') or c.startswith('category_') or c.startswith('location_') or c == 'loc_to_cat_ratio' or c.startswith('n_covisitors')],
        'G': [c for c in train_feats.columns if c.startswith('mobile_') or c == 'is_mobile_only' or c == 'contact_actions_ratio'],
        'H': [c for c in train_feats.columns if c.endswith('_rate') or c == 'consecutive_same_event_share']
    }
    
    # 3 фолда Walk-Forward CV
    dates = sorted(train_feats['date'].unique())
    wf_splits = [
        # Split 1: Train дни 0..6 (06.04-12.04, 7дн) -> Val дни 7..9 (13.04-15.04, 3дн)
        (train_feats['date'] <= dates[6], (train_feats['date'] >= dates[7]) & (train_feats['date'] <= dates[9])),
        # Split 2: Train дни 0..8 (06.04-14.04, 9дн) -> Val дни 9..11 (15.04-17.04, 3дн)
        (train_feats['date'] <= dates[8], (train_feats['date'] >= dates[9]) & (train_feats['date'] <= dates[11])),
        # Split 3: Train дни 0..10 (06.04-16.04, 11дн) -> Val дни 11..13 (17.04-19.04, 3дн)
        (train_feats['date'] <= dates[10], (train_feats['date'] >= dates[11]) & (train_feats['date'] <= dates[13])),
    ]
    
    # Primary Holdout: Неделя 1 -> Неделя 2
    is_holdout_val = train_feats['date'] >= dates[7]
    
    return train_feats, wf_splits, is_holdout_val, blocks


def evaluate_model(
    model_type: str,
    feature_cols: List[str],
    train_feats: pd.DataFrame,
    wf_splits: List[Tuple[np.ndarray, np.ndarray]],
    is_holdout_val: np.ndarray,
    model_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Универсальная функция оценки: обучает модель на Walk-Forward фолдах
    и основном Holdout, рассчитывает Bootstrap CI и recall по 4 архетипам.
    """
    cat_cols = [c for c in ['ua_os_family', 'ua_browser_family', 'platform_bucket'] if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    y_all = train_feats['target'].values
    
    # 1. Walk-Forward CV
    cv_scores = []
    for tr_mask, va_mask in wf_splits:
        X_tr = train_feats.loc[tr_mask, feature_cols].copy()
        y_tr = y_all[tr_mask]
        X_va = train_feats.loc[va_mask, feature_cols].copy()
        y_va = y_all[va_mask]
        
        preds_va = fit_predict(model_type, X_tr, y_tr, X_va, cat_cols, num_cols, model_params)
        prec = precision_at_recall(y_va, preds_va)
        cv_scores.append(prec)
        
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))
    
    # 2. Primary Holdout (Неделя 1 -> Неделя 2)
    X_ho_tr = train_feats.loc[~is_holdout_val, feature_cols].copy()
    y_ho_tr = y_all[~is_holdout_val]
    X_ho_va = train_feats.loc[is_holdout_val, feature_cols].copy()
    y_ho_va = y_all[is_holdout_val]
    clusters_va = train_feats.loc[is_holdout_val, 'bot_cluster'].values
    
    preds_ho = fit_predict(model_type, X_ho_tr, y_ho_tr, X_ho_va, cat_cols, num_cols, model_params)
    thresh, ho_prec, cl_recalls = get_threshold_and_cluster_recalls(y_ho_va, preds_ho, clusters_va)
    
    # 3. Bootstrap CI на Holdout
    ci_lower, ci_upper = compute_bootstrap_ci(y_ho_va, preds_ho, n_bootstraps=1000)
    
    return {
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'holdout_prec': ho_prec,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'rec_cl0': cl_recalls.get(0, 0.0),
        'rec_cl1': cl_recalls.get(1, 0.0),
        'rec_cl2': cl_recalls.get(2, 0.0),
        'rec_cl3': cl_recalls.get(3, 0.0),
        'threshold': thresh,
    }


def fit_predict(
    model_type: str, 
    X_tr: pd.DataFrame, 
    y_tr: np.ndarray, 
    X_va: pd.DataFrame, 
    cat_cols: List[str], 
    num_cols: List[str], 
    params: Dict[str, Any] = None
) -> np.ndarray:
    """Обучение и получение скоров вероятностей для выбранного семейства моделей."""
    params = params or {}
    
    if model_type == 'random':
        rng = np.random.RandomState(params.get('seed', 42))
        return rng.rand(len(X_va))
        
    elif model_type == 'rule_based':
        # Эвристический скор без ML
        # z(dt_cv) + z(-pointer_fill_rate_desktop) + z(search_page_max) + z(-cookie_age_days) + z(is_known_scraper_lib)
        score = np.zeros(len(X_va))
        for col, sign in [('dt_cv', 1), ('search_page_max', 1), ('is_known_scraper_lib', 1), 
                          ('pointer_fill_rate_desktop', -1), ('cookie_age_days', -1)]:
            if col in X_va.columns:
                vals = X_va[col].fillna(X_va[col].median()).values
                std = vals.std() + 1e-6
                z = (vals - vals.mean()) / std
                score += sign * z
        # Нормировка в [0, 1]
        score = (score - score.min()) / (score.max() - score.min() + 1e-6)
        return score
        
    elif model_type == 'logistic_regression':
        # Заполнение пропусков и масштабирование
        X_tr_num = X_tr[num_cols].fillna(X_tr[num_cols].median())
        X_va_num = X_va[num_cols].fillna(X_tr[num_cols].median())
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_num)
        X_va_sc = scaler.transform(X_va_num)
        
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_tr_sc, y_tr)
        return lr.predict_proba(X_va_sc)[:, 1]
        
    elif model_type == 'lightgbm':
        X_tr_lgb = X_tr.copy()
        X_va_lgb = X_va.copy()
        for c in cat_cols:
            X_tr_lgb[c] = X_tr_lgb[c].astype('category')
            X_va_lgb[c] = X_va_lgb[c].astype('category')
            
        p = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        p.update(params)
        model = LGBMClassifier(**p)
        model.fit(X_tr_lgb, y_tr)
        return model.predict_proba(X_va_lgb)[:, 1]
        
    elif model_type == 'catboost':
        X_tr_cb = X_tr.copy()
        X_va_cb = X_va.copy()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)
            
        p = {
            'iterations': 400,
            'learning_rate': 0.05,
            'depth': 6,
            'random_seed': 42,
            'verbose': 0,
            'thread_count': -1
        }
        p.update(params)
        model = CatBoostClassifier(**p, cat_features=cat_cols)
        model.fit(X_tr_cb, y_tr)
        return model.predict_proba(X_va_cb)[:, 1]
        
    elif model_type == 'xgboost':
        # One-hot кодирование категориальных колонок для XGBoost
        X_tr_xgb = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
        X_va_xgb = pd.get_dummies(X_va, columns=cat_cols, drop_first=True)
        X_va_xgb = X_va_xgb.reindex(columns=X_tr_xgb.columns, fill_value=0)
        
        p = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1
        }
        p.update(params)
        model = XGBClassifier(**p)
        model.fit(X_tr_xgb, y_tr)
        return model.predict_proba(X_va_xgb)[:, 1]
        
    elif model_type == 'random_forest':
        # Заполнение пропусков и one-hot для RF
        X_tr_rf = pd.get_dummies(X_tr, columns=cat_cols, drop_first=True)
        X_va_rf = pd.get_dummies(X_va, columns=cat_cols, drop_first=True)
        X_va_rf = X_va_rf.reindex(columns=X_tr_rf.columns, fill_value=0)
        
        X_tr_rf = X_tr_rf.fillna(-999.0)
        X_va_rf = X_va_rf.fillna(-999.0)
        
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
        rf.fit(X_tr_rf, y_tr)
        return rf.predict_proba(X_va_rf)[:, 1]
        
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")


def run_all_experiments() -> pd.DataFrame:
    """Запуск всей сетки экспериментов: Базовые уровни, Инкрементальная лестница, Ablation, Модели."""
    t_start = time.time()
    train_feats, wf_splits, is_holdout_val, blocks = prepare_data_and_clusters()
    
    results = []
    
    def log_result(exp_id: str, desc: str, model_name: str, res: Dict[str, Any]):
        ci_str = f"[{res['ci_lower']:.3f}, {res['ci_upper']:.3f}]"
        rec_str = f"{res['rec_cl0']:.1%}/{res['rec_cl1']:.1%}/{res['rec_cl2']:.1%}/{res['rec_cl3']:.1%}"
        print(f"[{exp_id:>2}] {desc:<35} | {model_name:<10} | CV: {res['cv_mean']:.4f}±{res['cv_std']:.4f} | Holdout: {res['holdout_prec']:.4f} {ci_str} | Recalls (0/1/2/3): {rec_str}")
        results.append({
            'ID': exp_id,
            'Конфигурация': desc,
            'Модель': model_name,
            'CV mean P@R>=0.7': round(res['cv_mean'], 4),
            'CV std': round(res['cv_std'], 4),
            'Holdout P@R>=0.7': round(res['holdout_prec'], 4),
            'Bootstrap 90% CI': ci_str,
            'Rec_Cl0 (Агрессивный)': round(res['rec_cl0'], 3),
            'Rec_Cl1 (HTTP-скрипт)': round(res['rec_cl1'], 3),
            'Rec_Cl2 (Мобильный)': round(res['rec_cl2'], 3),
            'Rec_Cl3 (Stealth-веб)': round(res['rec_cl3'], 3),
        })

    print("\n==================================================================")
    print("  ЭТАП 1: Базовые уровни (0 - 3)")
    print("==================================================================")
    
    # 0. Random Baseline
    res_rand = evaluate_model('random', blocks['A'], train_feats, wf_splits, is_holdout_val)
    log_result('0a', 'Random Baseline', 'Random', res_rand)
    
    # 1. Rule-based Heuristic
    res_rule = evaluate_model('rule_based', blocks['A'] + blocks['B'] + blocks['C'] + blocks['D'] + blocks['E'] + blocks['F'], 
                              train_feats, wf_splits, is_holdout_val)
    log_result('1', 'Rule-based Эвристика', 'Heuristic', res_rule)
    
    # 2. Logistic Regression (Top 12 фичей)
    lr_cols = [
        'n_events_total', 'cookie_age_days', 'dt_median_sec', 'dt_cv', 'night_activity_share',
        'pointer_fill_rate_desktop', 'is_known_scraper_lib', 'platform_os_mismatch',
        'search_page_max', 'loc_to_cat_ratio', 'is_mobile_only', 'share_events_seller_page_view'
    ]
    res_lr = evaluate_model('logistic_regression', lr_cols, train_feats, wf_splits, is_holdout_val)
    log_result('2', 'Logistic Regression (Top-12)', 'LogReg', res_lr)
    
    # 3. Официальный Baseline: LightGBM на Блоках A + B
    cols_ab = blocks['A'] + blocks['B']
    res_lgb_ab = evaluate_model('lightgbm', cols_ab, train_feats, wf_splits, is_holdout_val)
    log_result('3', 'Baseline: Блок A + B', 'LightGBM', res_lgb_ab)

    print("\n==================================================================")
    print("  ЭТАП 2: Инкрементальная лестница признаков (Forward Ladder)")
    print("==================================================================")
    
    current_cols = cols_ab.copy()
    ladder_steps = [
        ('+ Блок C (Временная динамика)', 'C'),
        ('+ Блок F (Каталог и пагинация)', 'F'),
        ('+ Блок D (Координаты мыши)', 'D'),
        ('+ Блок E (User-Agent и платформа)', 'E'),
        ('+ Блок G (Мобильные интеракции)', 'G'),
        ('+ Блок H (Последовательности)', 'H'),
    ]
    
    step_id = 4
    for step_name, blk_key in ladder_steps:
        current_cols.extend(blocks[blk_key])
        res_step = evaluate_model('lightgbm', current_cols, train_feats, wf_splits, is_holdout_val)
        log_result(str(step_id), step_name, 'LightGBM', res_step)
        step_id += 1

    full_cols = current_cols.copy()

    print("\n==================================================================")
    print("  ЭТАП 3: Контрольный Backward Ablation (Leave-One-Block-Out)")
    print("==================================================================")
    
    for blk_key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        cols_without = [c for c in full_cols if c not in blocks[blk_key]]
        res_abl = evaluate_model('lightgbm', cols_without, train_feats, wf_splits, is_holdout_val)
        log_result(f'-{blk_key}', f'Full без Блока {blk_key}', 'LightGBM', res_abl)

    print("\n==================================================================")
    print("  ЭТАП 4: Сравнение семейств моделей (на лучших фичах)")
    print("==================================================================")
    
    # 1. CatBoost
    res_cb = evaluate_model('catboost', full_cols, train_feats, wf_splits, is_holdout_val)
    log_result('M1', 'Full Features', 'CatBoost', res_cb)
    
    # 2. XGBoost
    res_xgb = evaluate_model('xgboost', full_cols, train_feats, wf_splits, is_holdout_val)
    log_result('M2', 'Full Features', 'XGBoost', res_xgb)
    
    # 3. Random Forest
    res_rf = evaluate_model('random_forest', full_cols, train_feats, wf_splits, is_holdout_val)
    log_result('M3', 'Full Features', 'RandForest', res_rf)
    
    # 4. Ансамбль (Бленд CatBoost + LightGBM)
    print("\n--- Оценка бленда (CatBoost 50% + LightGBM 50%) ---")
    cat_cols = ['ua_os_family', 'ua_browser_family', 'platform_bucket']
    X_ho_tr = train_feats.loc[~is_holdout_val, full_cols].copy()
    y_ho_tr = train_feats.loc[~is_holdout_val, 'target'].values
    X_ho_va = train_feats.loc[is_holdout_val, full_cols].copy()
    y_ho_va = train_feats.loc[is_holdout_val, 'target'].values
    clusters_va = train_feats.loc[is_holdout_val, 'bot_cluster'].values
    
    p_lgb = fit_predict('lightgbm', X_ho_tr, y_ho_tr, X_ho_va, cat_cols, [])
    p_cb = fit_predict('catboost', X_ho_tr, y_ho_tr, X_ho_va, cat_cols, [])
    p_blend = 0.5 * p_lgb + 0.5 * p_cb
    
    t_bl, ho_bl, cl_bl = get_threshold_and_cluster_recalls(y_ho_va, p_blend, clusters_va)
    ci_lo, ci_up = compute_bootstrap_ci(y_ho_va, p_blend, n_bootstraps=1000)
    
    # Blend CV
    cv_bl_scores = []
    for tr_mask, va_mask in wf_splits:
        X_tr = train_feats.loc[tr_mask, full_cols].copy()
        y_tr = train_feats.loc[tr_mask, 'target'].values
        X_va = train_feats.loc[va_mask, full_cols].copy()
        y_va = train_feats.loc[va_mask, 'target'].values
        p_l = fit_predict('lightgbm', X_tr, y_tr, X_va, cat_cols, [])
        p_c = fit_predict('catboost', X_tr, y_tr, X_va, cat_cols, [])
        cv_bl_scores.append(precision_at_recall(y_va, 0.5 * p_l + 0.5 * p_c))
        
    res_blend = {
        'cv_mean': float(np.mean(cv_bl_scores)),
        'cv_std': float(np.std(cv_bl_scores)),
        'holdout_prec': ho_bl,
        'ci_lower': ci_lo,
        'ci_upper': ci_up,
        'rec_cl0': cl_bl.get(0, 0.0),
        'rec_cl1': cl_bl.get(1, 0.0),
        'rec_cl2': cl_bl.get(2, 0.0),
        'rec_cl3': cl_bl.get(3, 0.0),
        'threshold': t_bl
    }
    log_result('ENS', 'Blend (50% LGBM + 50% CB)', 'Ensemble', res_blend)

    df_results = pd.DataFrame(results)
    df_results.to_csv('experiments_summary.csv', index=False)
    print(f"\n[OK] Все эксперименты завершены за {time.time() - t_start:.2f}s. Результаты сохранены в experiments_summary.csv")
    return df_results


if __name__ == '__main__':
    run_all_experiments()
