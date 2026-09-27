"""
features.py — Модуль препроцессинга и генерации признаков для детекции ботов.

Архитектура учитывает выводы двух раундов EDA и карту 4 архетипов ботов:
- Кластер 0: Агрессивный headless scraper (высокая скорость, глубокая пагинация)
- Кластер 1: Прямые HTTP-скрипты (Scrapy/curl/requests)
- Кластер 2: Мобильные эмуляторы / App парсеры (контакты/селлеры)
- Кластер 3: Stealth Web боты (маскировка под браузер с эмуляцией мыши)

Инварианты пайплайна:
1. Полное удаление дубликатов строк (без удаления параллельных событий).
2. Нормализация регистра platform -> platform.lower().
3. Хронологическая сортировка событий по (cookie_id, event_ts).
4. Строгая фильтрация по полуинтервалу: window_start_ts <= event_ts < window_end_ts.
5. Регрессионный тест на отсутствие captcha_shown внутри окна.
6. Явный Blacklist запрещенных признаков (постоконная активность, сырые даты).
"""

from __future__ import annotations

import sys
import re
import time
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

FORBIDDEN_KEYWORDS = ['captcha', 'after_window', 'days_until', 'raw_ts']


def preprocess_events(events_raw: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Шаг 0: Строгий препроцессинг-пайплайн.
    
    1. events_raw.drop_duplicates() — только полные дубликаты строк (4 863 шт.),
       НЕ удаляя параллельные события в одну секунду.
    2. platform = platform.str.lower() — нормализация регистра.
    3. sort_values(['cookie_id', 'event_ts']) — хронологическая сортировка.
    4. merge с meta_df по cookie_id.
    5. STRICT FILTER: window_start_ts <= event_ts < window_end_ts (полуинтервал).
    6. assert отсутствие captcha_shown.
    7. assert строгое попадание всех событий в [start, end).
    """
    print("--- [Препроцессинг] Запуск препроцессинга событий ---")
    n_raw = len(events_raw)
    
    # 1. Удаление полных дубликатов
    events = events_raw.drop_duplicates().copy()
    n_dedup = len(events)
    print(f"1. Удалено полных дубликатов строк: {n_raw - n_dedup:,}")
    
    # 2. Нормализация регистра платформ
    events['platform'] = events['platform'].str.lower()
    print("2. Регистр платформ приведен к нижнему регистру.")
    
    # 3. Хронологическая сортировка
    events = events.sort_values(['cookie_id', 'event_ts'])
    print("3. События отсортированы по (cookie_id, event_ts).")
    
    # 4. Объединение с метаданными окон
    meta_cols = ['cookie_id', 'cookie_created_at', 'window_start_ts', 'window_end_ts']
    if 'target' in meta_df.columns:
        meta_cols.append('target')
    
    events_merged = events.merge(meta_df[meta_cols], on='cookie_id', how='inner')
    
    # 5. Строгая фильтрация по полуинтервалу [start, end)
    events_filtered = events_merged[
        (events_merged['event_ts'] >= events_merged['window_start_ts']) & 
        (events_merged['event_ts'] < events_merged['window_end_ts'])
    ].copy()
    print(f"4-5. События отфильтрованы по полуинтервалу: {len(events_filtered):,} из {len(events_merged):,}")
    
    # 6. Регрессионный тест на отсутствие капчи внутри окна
    captcha_count = (events_filtered['event_name'] == 'captcha_shown').sum()
    assert captcha_count == 0, f"Критическая утечка: обнаружено {captcha_count} показов капчи внутри окна!"
    print("6. Регрессионный тест пройден: captcha_shown = 0.")
    
    # 7. Проверка граничных условий
    is_valid_range = events_filtered['event_ts'].between(
        events_filtered['window_start_ts'], 
        events_filtered['window_end_ts'], 
        inclusive='left'
    ).all()
    assert is_valid_range, "Нарушение временных границ окна наблюдения!"
    print("7. Проверка временных границ пройдена успешно.")
    
    return events_filtered


def compute_block_a_volume_structure(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Блок A: Объём и структура активности (baseline-агрегаты).
    Ловит: базовые объемы, конверсии и фокус на селлеров (Кластер 2).
    """
    g = ev.groupby('cookie_id')
    n_events_total = g.size().rename('n_events_total')
    n_events_log1p = np.log1p(n_events_total).rename('n_events_log1p')
    n_unique_event_names = g['event_name'].nunique().rename('n_unique_event_names')
    n_unique_item_id = g['item_id'].nunique().rename('n_unique_item_id')
    n_unique_search_query = g['search_query'].nunique().rename('n_unique_search_query')
    
    event_names = [
        'seller_page_view', 'contact_message_sent', 'contact_phone_show', 
        'item_view', 'search_results_view', 'favorite_add', 'login', 'photo_swipe', 'contact_chat_open'
    ]
    event_counts = ev.groupby(['cookie_id', 'event_name']).size().unstack(fill_value=0)
    for en in event_names:
        if en not in event_counts.columns:
            event_counts[en] = 0
            
    event_counts = event_counts[event_names]
    event_shares = event_counts.div(n_events_total, axis=0)
    
    event_counts.columns = [f'n_events_{c}' for c in event_counts.columns]
    event_shares.columns = [f'share_events_{c}' for c in event_shares.columns]
    
    item_view_to_search_ratio = (
        event_counts['n_events_item_view'] / np.maximum(event_counts['n_events_search_results_view'], 1.0)
    ).rename('item_view_to_search_ratio')
    
    login_flag = (event_counts['n_events_login'] > 0).astype(int).rename('login_flag')
    
    feats_a = pd.concat([
        n_events_total,
        n_events_log1p,
        n_unique_event_names,
        n_unique_item_id,
        n_unique_search_query,
        event_counts,
        event_shares,
        item_view_to_search_ratio,
        login_flag,
    ], axis=1).reset_index()
    
    return feats_a


def compute_block_b_cookie_age(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Блок B: Возраст куки на момент начала окна наблюдения.
    Ловит: новореги и сброшенные identity (Кластер 3 и Кластер 0).
    """
    age_hours = (meta_df['window_start_ts'] - meta_df['cookie_created_at']).dt.total_seconds() / 3600.0
    age_days = age_hours / 24.0
    
    feats_b = pd.DataFrame({
        'cookie_id': meta_df['cookie_id'],
        'cookie_age_hours': age_hours,
        'cookie_age_days': age_days,
        'is_fresh_cookie_1d': (age_days < 1.0).astype(int),
        'is_fresh_cookie_7d': (age_days < 7.0).astype(int),
        'cookie_age_log': np.log1p(np.maximum(age_hours, 0.0)),
    })
    return feats_b


def compute_block_c_temporal_dynamics(ev: pd.DataFrame, n_events_total_series: pd.Series) -> pd.DataFrame:
    """
    Блок C: Временная динамика: интервалы Δt, циркадные ритмы и структура сессии.
    Ловит: сверхбыстрых ботов (Кластер 0, dt~13с) и регулярность скриптов (Кластер 1 и 3).
    """
    # 1. Интервалы дельт времени
    ev['dt'] = ev.groupby('cookie_id')['event_ts'].diff().dt.total_seconds()
    ev['hour'] = ev['event_ts'].dt.hour
    
    dt_agg = ev.groupby('cookie_id')['dt'].agg(
        dt_median_sec='median',
        dt_mean_sec='mean',
        dt_std_sec='std',
        dt_min_sec='min',
        share_instant_actions=lambda x: (x < 1.0).mean(),
        share_dt_under_5s=lambda x: (x < 5.0).mean(),
        share_dt_under_10s=lambda x: (x < 10.0).mean(),
        dt_p10=lambda x: x.quantile(0.10) if len(x.dropna()) > 0 else np.nan,
        dt_p90=lambda x: x.quantile(0.90) if len(x.dropna()) > 0 else np.nan,
    ).reset_index()
    
    dt_agg['dt_cv'] = dt_agg['dt_std_sec'] / (dt_agg['dt_mean_sec'] + 1e-6)
    
    # 2. Векторизованная структура сессии
    session_bounds = ev.groupby('cookie_id').agg(
        first_ts=('event_ts', 'first'),
        last_ts=('event_ts', 'last'),
        win_start=('window_start_ts', 'first'),
        win_end=('window_end_ts', 'first'),
    ).reset_index()
    
    session_bounds['session_span_sec'] = (session_bounds['last_ts'] - session_bounds['first_ts']).dt.total_seconds()
    session_bounds['session_span_ratio'] = session_bounds['session_span_sec'] / 86400.0
    session_bounds['time_to_first_event_sec'] = (session_bounds['first_ts'] - session_bounds['win_start']).dt.total_seconds()
    session_bounds['time_from_last_event_to_end_sec'] = (session_bounds['win_end'] - session_bounds['last_ts']).dt.total_seconds()
    
    # 3. Векторизованный 24-часовой суточный профиль через crosstab
    hour_crosstab = pd.crosstab(ev['cookie_id'], ev['hour']).reindex(columns=range(24), fill_value=0)
    tot_per_cookie = hour_crosstab.sum(axis=1)
    hour_probs = hour_crosstab.div(tot_per_cookie, axis=0)
    
    h_p = hour_probs.values
    nz_mask = h_p > 0
    entropy_vals = -np.sum(np.where(nz_mask, h_p * np.log2(h_p + 1e-12), 0.0), axis=1)
    
    night_counts = hour_crosstab[[0, 1, 2, 3, 4, 5, 6]].sum(axis=1)
    circ_df = pd.DataFrame({
        'cookie_id': hour_crosstab.index,
        'night_activity_share': (night_counts / tot_per_cookie).values,
        'active_hours_count': (hour_crosstab > 0).sum(axis=1).values,
        'peak_hour_share': hour_probs.max(axis=1).values,
        'hourly_entropy': entropy_vals,
        'hourly_cv': (hour_crosstab.std(axis=1) / (hour_crosstab.mean(axis=1) + 1e-6)).values,
    })
    
    circ_sess = session_bounds[[
        'cookie_id', 'session_span_sec', 'session_span_ratio', 
        'time_to_first_event_sec', 'time_from_last_event_to_end_sec'
    ]].merge(circ_df, on='cookie_id')
    
    circ_sess = circ_sess.merge(n_events_total_series.reset_index(), on='cookie_id')
    circ_sess['events_per_active_hour'] = circ_sess['n_events_total'] / np.maximum(circ_sess['active_hours_count'], 1.0)
    circ_sess = circ_sess.drop(columns=['n_events_total'])
    
    feats_c = dt_agg.merge(circ_sess, on='cookie_id')
    return feats_c


def compute_block_d_mouse_pointer(ev: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Блок D: Координаты курсора мыши, строго сегментированные по десктопу/вебу.
    На мобильных курсор отсутствует объективно, поэтому считаем только по desktop/web.
    Ловит: headless-ботов (Кластер 0) и эмуляцию мыши (Кластер 3).
    """
    is_desktop = ev['platform'].isin(['desktop', 'web'])
    ev_desk = ev[is_desktop].copy()
    
    desk_counts = ev_desk.groupby('cookie_id').size().rename('desktop_events_count')
    ev_desk_ptr = ev_desk[ev_desk['pointer_x'].notna()].copy()
    ptr_counts = ev_desk_ptr.groupby('cookie_id').size().rename('pointer_events_count')
    
    # Перемещения между последовательными координатами курсора
    ev_desk_ptr['dx'] = ev_desk_ptr.groupby('cookie_id')['pointer_x'].diff()
    ev_desk_ptr['dy'] = ev_desk_ptr.groupby('cookie_id')['pointer_y'].diff()
    ev_desk_ptr['step'] = np.sqrt(ev_desk_ptr['dx']**2 + ev_desk_ptr['dy']**2)
    
    ptr_step_mean = ev_desk_ptr.groupby('cookie_id')['step'].mean().rename('pointer_step_mean')
    ptr_x_std = ev_desk_ptr.groupby('cookie_id')['pointer_x'].std().rename('pointer_x_std')
    ptr_y_std = ev_desk_ptr.groupby('cookie_id')['pointer_y'].std().rename('pointer_y_std')
    
    ev_desk_ptr['point_key'] = ev_desk_ptr['pointer_x'].astype(str) + '_' + ev_desk_ptr['pointer_y'].astype(str)
    ptr_unique_count = ev_desk_ptr.groupby('cookie_id')['point_key'].nunique().rename('pointer_unique_count')
    
    feats_d = meta_df[['cookie_id']].merge(desk_counts.reset_index(), on='cookie_id', how='left')
    feats_d['desktop_events_count'] = feats_d['desktop_events_count'].fillna(0).astype(int)
    feats_d['has_desktop_events'] = (feats_d['desktop_events_count'] > 0).astype(int)
    
    feats_d = feats_d.merge(ptr_counts.reset_index(), on='cookie_id', how='left')
    feats_d['pointer_events_count'] = feats_d['pointer_events_count'].fillna(0).astype(int)
    feats_d['pointer_fill_rate_desktop'] = np.where(
        feats_d['desktop_events_count'] > 0,
        feats_d['pointer_events_count'] / feats_d['desktop_events_count'],
        np.nan
    )
    
    feats_d = feats_d.merge(ptr_x_std.reset_index(), on='cookie_id', how='left')
    feats_d = feats_d.merge(ptr_y_std.reset_index(), on='cookie_id', how='left')
    feats_d = feats_d.merge(ptr_unique_count.reset_index(), on='cookie_id', how='left')
    feats_d['pointer_unique_ratio'] = np.where(
        feats_d['pointer_events_count'] > 0,
        feats_d['pointer_unique_count'] / feats_d['pointer_events_count'],
        np.nan
    )
    feats_d = feats_d.merge(ptr_step_mean.reset_index(), on='cookie_id', how='left')
    feats_d = feats_d.drop(columns=['pointer_unique_count'])
    return feats_d


def compute_block_e_user_agent_platform(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Блок E: Парсинг User-Agent и согласованность с Platform.
    Ловит: прямые библиотеки (Кластер 1) и спуфинг ОС/платформы.
    """
    ua_series = ev['user_agent'].astype(str)
    ua_lower = ua_series.str.lower()
    
    lib_pattern = r'curl|scrapy|python|requests|urllib|go-http-client|node-fetch'
    ev['is_known_scraper_lib'] = ua_lower.str.contains(lib_pattern, regex=True).astype(int)
    ev['is_headless_ua'] = ua_lower.str.contains('headless').astype(int)
    
    def get_os_family(ua: str) -> str:
        if 'windows' in ua:
            return 'Windows'
        if 'macintosh' in ua or 'mac os' in ua:
            return 'Mac'
        if 'android' in ua:
            return 'Android'
        if 'iphone' in ua or 'ipad' in ua or 'ios' in ua:
            return 'iOS'
        if 'linux' in ua:
            return 'Linux'
        return 'Other'
        
    def get_browser_family(ua: str) -> str:
        if 'chrome' in ua or 'crios' in ua:
            if 'edg' in ua or 'edge' in ua:
                return 'Edge'
            if 'opr' in ua or 'opera' in ua:
                return 'Opera'
            return 'Chrome'
        if 'safari' in ua and 'android' not in ua:
            return 'Safari'
        if 'firefox' in ua or 'fxios' in ua:
            return 'Firefox'
        if re.search(lib_pattern, ua):
            return 'Script_Bot'
        return 'Other'
        
    def get_platform_bucket(p: str) -> str:
        if p in ['desktop', 'web']:
            return 'desktop_web'
        if p in ['android']:
            return 'android'
        if p in ['ios', 'iphone']:
            return 'ios'
        return 'other'

    ev['ua_os_family'] = ua_lower.apply(get_os_family)
    ev['ua_browser_family'] = ua_lower.apply(get_browser_family)
    ev['platform_bucket'] = ev['platform'].apply(get_platform_bucket)
    
    is_mob_plat = ev['platform_bucket'].isin(['android', 'ios'])
    is_desk_plat = ev['platform_bucket'] == 'desktop_web'
    is_mob_os = ev['ua_os_family'].isin(['Android', 'iOS'])
    is_desk_os = ev['ua_os_family'].isin(['Windows', 'Mac', 'Linux'])
    ev['platform_os_mismatch'] = ((is_mob_plat & is_desk_os) | (is_desk_plat & is_mob_os)).astype(int)
    
    cookie_ua = ev.groupby('cookie_id').agg(
        is_known_scraper_lib=('is_known_scraper_lib', 'max'),
        is_headless_ua=('is_headless_ua', 'max'),
        ua_nunique_within_cookie=('user_agent', 'nunique'),
        platform_nunique_within_cookie=('platform', 'nunique'),
        platform_os_mismatch=('platform_os_mismatch', 'max'),
        ua_os_family=('ua_os_family', lambda s: s.mode().iloc[0] if len(s)>0 else 'Other'),
        ua_browser_family=('ua_browser_family', lambda s: s.mode().iloc[0] if len(s)>0 else 'Other'),
        platform_bucket=('platform_bucket', lambda s: s.mode().iloc[0] if len(s)>0 else 'other')
    ).reset_index()
    
    cookie_ua['ua_changed_flag'] = (cookie_ua['ua_nunique_within_cookie'] > 1).astype(int)
    cookie_ua['platform_changed_flag'] = (cookie_ua['platform_nunique_within_cookie'] > 1).astype(int)
    return cookie_ua


def compute_block_f_catalog_pagination_covisitation(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Блок F: Каталоговое поведение, глубина пагинации, разнообразие и со-визитация.
    Ловит: экстремальный обход листинга (Кластер 0, page_max~21).
    """
    ev['hour_bucket'] = ev['event_ts'].dt.floor('h')
    
    # 1. Со-визитация (по часовым бакетам и общий пул объявлений)
    ev_items = ev[ev['item_id'].notna()].copy()
    hourly_covisitors = ev_items.groupby(['item_id', 'hour_bucket'])['cookie_id'].nunique().rename('covisitors_hourly')
    ev = ev.merge(hourly_covisitors.reset_index(), on=['item_id', 'hour_bucket'], how='left')
    
    total_covisitors = ev_items.groupby('item_id')['cookie_id'].nunique().rename('covisitors_total')
    ev = ev.merge(total_covisitors.reset_index(), on='item_id', how='left')
    
    # 2. Монотонность пагинации
    ev_search = ev[ev['search_page'].notna()].copy()
    ev_search['page_diff'] = ev_search.groupby('cookie_id')['search_page'].diff()
    mono_counts = (ev_search['page_diff'] >= 0).groupby(ev_search['cookie_id']).sum()
    pair_counts = ev_search['page_diff'].notna().groupby(ev_search['cookie_id']).sum()
    mono_shares = (mono_counts / np.maximum(pair_counts, 1.0)).rename('search_page_monotonic_share').reset_index()
    
    cookie_cat_loc = ev.groupby('cookie_id').agg(
        search_page_max=('search_page', 'max'),
        search_page_mean=('search_page', 'mean'),
        category_nunique=('item_category', 'nunique'),
        location_nunique=('item_location', 'nunique'),
        n_covisitors_mean=('covisitors_hourly', 'mean'),
        n_covisitors_max=('covisitors_hourly', 'max'),
        n_covisitors_total_mean=('covisitors_total', 'mean'),
        n_covisitors_total_max=('covisitors_total', 'max'),
    ).reset_index()
    
    cookie_cat_loc['search_page_max'] = cookie_cat_loc['search_page_max'].fillna(0.0)
    cookie_cat_loc['search_page_mean'] = cookie_cat_loc['search_page_mean'].fillna(0.0)
    cookie_cat_loc['n_covisitors_mean'] = cookie_cat_loc['n_covisitors_mean'].fillna(1.0)
    cookie_cat_loc['n_covisitors_max'] = cookie_cat_loc['n_covisitors_max'].fillna(1.0)
    cookie_cat_loc['n_covisitors_total_mean'] = cookie_cat_loc['n_covisitors_total_mean'].fillna(1.0)
    cookie_cat_loc['n_covisitors_total_max'] = cookie_cat_loc['n_covisitors_total_max'].fillna(1.0)
    cookie_cat_loc['loc_to_cat_ratio'] = cookie_cat_loc['location_nunique'] / np.maximum(cookie_cat_loc['category_nunique'], 1.0)
    
    feats_f = cookie_cat_loc.merge(mono_shares, on='cookie_id', how='left')
    feats_f['search_page_monotonic_share'] = feats_f['search_page_monotonic_share'].fillna(0.0)
    return feats_f


def compute_block_g_mobile_interactions(ev: pd.DataFrame, feats_a: pd.DataFrame) -> pd.DataFrame:
    """
    Блок G: Специфические интеракции мобильных парсеров (Кластер 2).
    Ловит: автоматизированный парсинг мобильных приложений/API без мыши.
    """
    is_mob_event = ev['platform'].isin(['android', 'ios', 'iphone'])
    mob_ev_count = is_mob_event.groupby(ev['cookie_id']).sum().rename('mob_events_count')
    tot_ev_count = ev.groupby('cookie_id').size().rename('tot_events')
    
    mob_df = pd.concat([mob_ev_count, tot_ev_count], axis=1).reset_index()
    mob_df['is_mobile_only'] = (mob_df['mob_events_count'] == mob_df['tot_events']).astype(int)
    
    mob_search_count = (is_mob_event & (ev['event_name'] == 'search_results_view')).groupby(ev['cookie_id']).sum().rename('mob_search_count')
    mob_df = mob_df.merge(mob_search_count.reset_index(), on='cookie_id', how='left')
    mob_df['mob_search_count'] = mob_df['mob_search_count'].fillna(0)
    mob_df['mobile_search_share'] = np.where(
        mob_df['mob_events_count'] > 0,
        mob_df['mob_search_count'] / mob_df['mob_events_count'],
        0.0
    )
    
    feats_g = mob_df[['cookie_id', 'is_mobile_only', 'mobile_search_share']].merge(
        feats_a[['cookie_id', 'share_events_seller_page_view', 'share_events_contact_phone_show', 'share_events_contact_message_sent']], 
        on='cookie_id'
    )
    feats_g['mobile_seller_focus_ratio'] = feats_g['share_events_seller_page_view'] * feats_g['is_mobile_only']
    feats_g['contact_actions_ratio'] = feats_g['share_events_contact_phone_show'] + feats_g['share_events_contact_message_sent']
    feats_g = feats_g[['cookie_id', 'is_mobile_only', 'mobile_search_share', 'mobile_seller_focus_ratio', 'contact_actions_ratio']]
    return feats_g


def compute_block_h_sequence_transitions(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Блок H: Последовательности действий и переходы между типами событий.
    Ловит: паттерны поиска и воронки взаимодействия.
    """
    ev['prev_event_name'] = ev.groupby('cookie_id')['event_name'].shift(1)
    ev['is_consecutive_same_event'] = (ev['event_name'] == ev['prev_event_name']).astype(int)
    ev['is_search_to_search'] = ((ev['prev_event_name'] == 'search_results_view') & (ev['event_name'] == 'search_results_view')).astype(int)
    ev['is_search_to_item'] = ((ev['prev_event_name'] == 'search_results_view') & (ev['event_name'] == 'item_view')).astype(int)
    ev['is_item_to_seller'] = ((ev['prev_event_name'] == 'item_view') & (ev['event_name'] == 'seller_page_view')).astype(int)
    
    seq_agg = ev.groupby('cookie_id').agg(
        consecutive_same_event_share=('is_consecutive_same_event', 'mean'),
        search_to_search_rate=('is_search_to_search', 'mean'),
        search_to_item_rate=('is_search_to_item', 'mean'),
        item_to_seller_rate=('is_item_to_seller', 'mean')
    ).reset_index()
    
    return seq_agg


def extract_all_features(
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    events_raw: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Основная точка входа: сквозное извлечение признаков для Train и Test.
    """
    start_time = time.time()
    print("==================================================================")
    print("  [+] Запуск полного пайплайна Feature Engineering (Блоки 0 - H)")
    print("==================================================================")
    
    meta_all = pd.concat([
        train_df[['cookie_id', 'cookie_created_at', 'window_start_ts', 'window_end_ts']],
        test_df[['cookie_id', 'cookie_created_at', 'window_start_ts', 'window_end_ts']]
    ], ignore_index=True)
    
    # Шаг 0: Препроцессинг
    ev_filtered = preprocess_events(events_raw, meta_all)
    
    # Блок A
    t_blk = time.time()
    feats_a = compute_block_a_volume_structure(ev_filtered)
    print(f"[OK] Блок A (Объём и структура) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок B
    t_blk = time.time()
    feats_b = compute_block_b_cookie_age(meta_all)
    print(f"[OK] Блок B (Возраст куки) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок C
    t_blk = time.time()
    feats_c = compute_block_c_temporal_dynamics(ev_filtered, feats_a.set_index('cookie_id')['n_events_total'])
    print(f"[OK] Блок C (Временная динамика и циркадные ритмы) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок D
    t_blk = time.time()
    feats_d = compute_block_d_mouse_pointer(ev_filtered, meta_all)
    print(f"[OK] Блок D (Координаты мыши на Desktop/Web) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок E
    t_blk = time.time()
    feats_e = compute_block_e_user_agent_platform(ev_filtered)
    print(f"[OK] Блок E (User-Agent и несогласованность с Platform) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок F
    t_blk = time.time()
    feats_f = compute_block_f_catalog_pagination_covisitation(ev_filtered)
    print(f"[OK] Блок F (Пагинация, каталог и со-визитация) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок G
    t_blk = time.time()
    feats_g = compute_block_g_mobile_interactions(ev_filtered, feats_a)
    print(f"[OK] Блок G (Интеракции мобильных парсеров) готов [{time.time() - t_blk:.2f}s]")
    
    # Блок H
    t_blk = time.time()
    feats_h = compute_block_h_sequence_transitions(ev_filtered)
    print(f"[OK] Блок H (Переходы и последовательности действий) готов [{time.time() - t_blk:.2f}s]")
    
    # Объединение всех блоков
    print("\n--- Объединение блоков признаков ---")
    all_features = meta_all[['cookie_id']].copy()
    blocks = [feats_a, feats_b, feats_c, feats_d, feats_e, feats_f, feats_g, feats_h]
    
    for b_df in blocks:
        all_features = all_features.merge(b_df, on='cookie_id', how='left')
        
    # Проверка черного списка признаков (Explicit Blacklist)
    cols = all_features.columns.tolist()
    for col in cols:
        for forbidden in FORBIDDEN_KEYWORDS:
            assert forbidden not in col.lower(), f"Запрещенный признак обнаружен в датасете: {col}!"
            
    print(f"Проверка Blacklist пройдена. Всего создано фичей: {len(cols) - 1}")
    
    # Разделение на train и test
    train_feats = train_df[['cookie_id', 'target']].merge(all_features, on='cookie_id', how='left')
    test_feats = test_df[['cookie_id']].merge(all_features, on='cookie_id', how='left')
    
    assert len(train_feats) == len(train_df), "Ошибка размера train_feats!"
    assert len(test_feats) == len(test_df), "Ошибка размера test_feats!"
    
    print(f"Итоговая размерность Train: {train_feats.shape}")
    print(f"Итоговая размерность Test:  {test_feats.shape}")
    print(f"Полное время генерации признаков: {time.time() - start_time:.2f}s")
    
    return train_feats, test_feats


def run_unit_tests(train_features: pd.DataFrame) -> None:
    """
    Unit-тесты на выборке из 4 ключевых архетипов ботов.
    Сверяем значения признаков с физическим смыслом кластеров.
    """
    print("\n==================================================================")
    print("  [TEST] Запуск Unit-тестов для 4 архетипов ботов")
    print("==================================================================")
    
    # Куки-представители из 4 кластеров
    samples = {
        'Кластер 0 (Агрессивный Scraper)': 'ck_0252050372ee3ce9',
        'Кластер 1 (HTTP-скрипт Scrapy/curl)': 'ck_012587ae7bc0bffe',
        'Кластер 2 (Мобильный эмулятор/App)': 'ck_00389b209b37394b',
        'Кластер 3 (Stealth Web бот с мышью)': 'ck_00708a91ae901fee',
    }
    
    for cluster_title, cid in samples.items():
        row = train_features[train_features['cookie_id'] == cid].iloc[0]
        print(f"\nТестируем {cluster_title} [cookie: {cid}]:")
        print(f"  - Всего событий (n_events_total): {row['n_events_total']}")
        print(f"  - Медиана дельты времени (dt_median_sec): {row['dt_median_sec']:.1f} сек")
        print(f"  - Заполнение мыши десктоп (pointer_fill_rate_desktop): {row['pointer_fill_rate_desktop']}")
        print(f"  - Максимальная страница поиска (search_page_max): {row['search_page_max']}")
        print(f"  - Флаг скриптовой библиотеки (is_known_scraper_lib): {row['is_known_scraper_lib']}")
        print(f"  - Мобильная кука (is_mobile_only): {row['is_mobile_only']}")
        print(f"  - Доля seller_page_view: {row['share_events_seller_page_view']:.3f}")
        
        if 'Кластер 0' in cluster_title:
            assert row['search_page_max'] >= 15.0, "Кластер 0 должен иметь глубокую пагинацию!"
            assert row['dt_median_sec'] < 25.0, "Кластер 0 должен быть сверхбыстрым!"
        elif 'Кластер 1' in cluster_title:
            assert row['is_known_scraper_lib'] == 1, "Кластер 1 должен идентифицироваться как скрипт!"
        elif 'Кластер 2' in cluster_title:
            assert row['is_mobile_only'] == 1, "Кластер 2 должен быть мобильным!"
            assert np.isnan(row['pointer_fill_rate_desktop']), "У мобильного бота не должно быть десктопной мыши!"
        elif 'Кластер 3' in cluster_title:
            assert row['pointer_fill_rate_desktop'] > 0.5, "Кластер 3 должен эмулировать мышь!"
            
        print("  -> Все утверждения для архетипа подтверждены!")
        
    print("\n[OK] Все Unit-тесты успешно пройдены!")


if __name__ == '__main__':
    train = pd.read_csv('data/train.csv', parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'])
    test = pd.read_csv('data/test.csv', parse_dates=['cookie_created_at', 'window_start_ts', 'window_end_ts'])
    events = pd.read_csv('data/events.csv.gz', parse_dates=['event_ts'])
    
    feats_train, feats_test = extract_all_features(train, test, events)
    run_unit_tests(feats_train)
    
    print("\nСохранение датасетов с признаками...")
    # Сохраняем в Parquet (быстро и с типами) и CSV
    feats_train.to_parquet('data/features_train.parquet', index=False)
    feats_test.to_parquet('data/features_test.parquet', index=False)
    feats_train.to_csv('data/features_train.csv', index=False)
    feats_test.to_csv('data/features_test.csv', index=False)
    print("[OK] Файлы успешно сохранены в data/features_train.parquet и data/features_test.parquet!")
