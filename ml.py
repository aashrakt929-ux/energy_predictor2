"""منطق التحليل والتنبؤ باستهلاك الكهرباء (Pandas + Scikit-learn).

الخطوات:
1) تجميع القراءات إلى استهلاك شهري.
2) نموذج Ridge Regression على (اتجاه زمني + موسمية شهرية Fourier).
3) اكتشاف الشذوذ بـ Robust Z-score على بواقي النموذج، ثم إعادة التدريب بدون الشواذ.
4) توقع الشهور القادمة مع نطاق ثقة.
5) مقارنة الشهور واقتراح طرق التوفير.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MONTHS_AR = [
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
]
SUMMER_MONTHS = {6, 7, 8, 9}
MIN_MONTHS = 3
MIN_PCT_ABOVE = 15.0  # لا نُنبّه لو الزيادة عن المتوقع أقل من 15% حتى لو إحصائيًا شاذة


# ---------------------------------------------------------------- تجهيز البيانات
def to_monthly(df: pd.DataFrame) -> tuple[pd.Series, int]:
    """يحوّل قراءات (date, kwh) يومية أو شهرية إلى سلسلة شهرية مرتبة.

    الشهور الناقصة داخل المدى تُملأ بالاستيفاء الخطي ويُرجَع عددها.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["kwh"] = pd.to_numeric(d["kwh"], errors="coerce")
    d = d.dropna(subset=["date", "kwh"])
    d = d[d["kwh"] >= 0]
    if d.empty:
        raise ValueError("لا توجد بيانات صالحة")
    monthly = d.groupby(d["date"].dt.to_period("M"))["kwh"].sum().sort_index()
    full = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(full)
    missing = int(monthly.isna().sum())
    monthly = monthly.interpolate(limit_direction="both")
    return monthly, missing


def _n_harmonics(n: int) -> int:
    if n >= 24:
        return 2
    if n >= 12:
        return 1
    return 0  # بيانات قليلة: اتجاه فقط بدون موسمية


def _design(periods, n_harm: int) -> np.ndarray:
    ords = np.array([p.ordinal for p in periods], dtype=float)
    months = np.array([p.month for p in periods], dtype=float)
    cols = [ords]
    for k in range(1, n_harm + 1):
        cols.append(np.sin(2 * np.pi * k * months / 12))
        cols.append(np.cos(2 * np.pi * k * months / 12))
    return np.column_stack(cols)


def _fit(X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None):
    model = make_pipeline(StandardScaler(), Ridge(alpha=0.5))
    model.fit(X, y, ridge__sample_weight=weights)
    return model


def _robust_sigma(resid: np.ndarray, floor: float) -> float:
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
    return float(max(mad, floor))


def _fit_robust(periods, y: np.ndarray, z_threshold: float):
    """تدريب على مرحلتين: كشف الشواذ ثم إعادة التدريب بدونها."""
    n_harm = _n_harmonics(len(y))
    X = _design(periods, n_harm)
    floor = 0.03 * float(np.mean(y)) or 1.0

    first = _fit(X, y)
    r1 = y - first.predict(X)
    z1 = (r1 - np.median(r1)) / _robust_sigma(r1, floor)
    outlier = np.abs(z1) >= z_threshold if len(y) >= 6 else np.zeros(len(y), bool)
    if outlier.sum() > len(y) // 3:  # لا نحذف أكثر من ثلث البيانات
        outlier = np.zeros(len(y), bool)

    weights = np.where(outlier, 0.0, 1.0)
    model = _fit(X, y, weights)
    resid = y - model.predict(X)
    inliers = resid[~outlier] if (~outlier).sum() > 2 else resid
    sigma = float(max(inliers.std(ddof=1) if len(inliers) > 2 else 0.0, floor))
    return model, n_harm, sigma


# ---------------------------------------------------------------- التحليل الرئيسي
def analyze(df: pd.DataFrame, horizon: int = 3, z_threshold: float = 2.5) -> dict:
    monthly, missing = to_monthly(df)
    n = len(monthly)
    if n < MIN_MONTHS:
        raise ValueError(f"محتاج {MIN_MONTHS} شهور على الأقل من البيانات (المتاح: {n})")

    periods = list(monthly.index)
    y = monthly.to_numpy(dtype=float)
    labels = [str(p) for p in periods]

    model, n_harm, sigma = _fit_robust(periods, y, z_threshold)
    X = _design(periods, n_harm)
    fitted = np.clip(model.predict(X), 0, None)
    z = (y - fitted) / sigma

    # --- الشذوذ (ارتفاع غير معتاد فقط): إحصائيًا (Z) وعمليًا (نسبة الزيادة)
    pct = np.where(fitted > 0, (y / np.where(fitted > 0, fitted, 1) - 1) * 100, 0.0)
    is_anom = (z >= z_threshold) & (pct >= MIN_PCT_ABOVE) if n >= 6 else np.zeros(n, bool)
    anomalies = [
        {
            "month": labels[i],
            "kwh": round(float(y[i]), 1),
            "expected": round(float(fitted[i]), 1),
            "pct_above": round(float(pct[i]), 1),
            "z": round(float(z[i]), 2),
        }
        for i in np.where(is_anom)[0]
    ]

    # --- التوقع
    future = [periods[-1] + k for k in range(1, horizon + 1)]
    pred = np.clip(model.predict(_design(future, n_harm)), 0, None)
    margin = 1.96 * sigma
    forecast = [
        {
            "month": str(p),
            "kwh": round(float(v), 1),
            "lower": round(float(max(v - margin, 0)), 1),
            "upper": round(float(v + margin), 1),
        }
        for p, v in zip(future, pred)
    ]

    # --- جودة النموذج (Backtest على آخر 3 شهور)
    backtest = None
    if n >= 18:
        tr, te = slice(0, n - 3), slice(n - 3, n)
        m_bt, h_bt, _ = _fit_robust(periods[: n - 3], y[tr], z_threshold)
        p_bt = np.clip(m_bt.predict(_design(periods[te], h_bt)), 0, None)
        mae = float(np.mean(np.abs(y[te] - p_bt)))
        mape = float(np.mean(np.abs(y[te] - p_bt) / np.where(y[te] == 0, 1, y[te])) * 100)
        backtest = {"mae": round(mae, 1), "mape": round(mape, 1)}

    # --- مقارنة الشهور
    monthly_table = []
    for i, p in enumerate(periods):
        mom = (y[i] / y[i - 1] - 1) * 100 if i >= 1 and y[i - 1] > 0 else None
        yoy = (y[i] / y[i - 12] - 1) * 100 if i >= 12 and y[i - 12] > 0 else None
        monthly_table.append({
            "month": labels[i],
            "label": f"{MONTHS_AR[p.month - 1]} {p.year}",
            "kwh": round(float(y[i]), 1),
            "expected": round(float(fitted[i]), 1),
            "mom": None if mom is None else round(float(mom), 1),
            "yoy": None if yoy is None else round(float(yoy), 1),
            "anomaly": bool(is_anom[i]),
        })

    by_year: dict[str, list] = {}
    for p, v in zip(periods, y):
        by_year.setdefault(str(p.year), [None] * 12)[p.month - 1] = round(float(v), 1)

    # --- إحصائيات
    imax, imin = int(np.argmax(y)), int(np.argmin(y))
    stats = {
        "months_count": n,
        "missing_filled": missing,
        "total": round(float(y.sum()), 1),
        "average": round(float(y.mean()), 1),
        "max": {"month": labels[imax], "kwh": round(float(y[imax]), 1)},
        "min": {"month": labels[imin], "kwh": round(float(y[imin]), 1)},
        "last": {"month": labels[-1], "kwh": round(float(y[-1]), 1)},
        "yearly_growth_pct": None,
        "peak_calendar_months": [],
    }
    if n >= 24:
        last12, prev12 = y[-12:].sum(), y[-24:-12].sum()
        if prev12 > 0:
            stats["yearly_growth_pct"] = round(float((last12 / prev12 - 1) * 100), 1)
    if n >= 12:
        cal = pd.Series(y, index=[p.month for p in periods]).groupby(level=0).mean()
        stats["peak_calendar_months"] = [int(m) for m in cal.nlargest(3).index]

    result = {
        "stats": stats,
        "monthly": monthly_table,
        "by_year": by_year,
        "forecast": forecast,
        "anomalies": anomalies,
        "backtest": backtest,
        "model": {"type": "Ridge Regression (trend + seasonality)", "harmonics": n_harm},
    }
    result["tips"] = suggest_tips(result)
    return result


# ---------------------------------------------------------------- الاقتراحات
def suggest_tips(result: dict) -> list[dict]:
    tips: list[dict] = []
    stats, anomalies, forecast = result["stats"], result["anomalies"], result["forecast"]

    if anomalies:
        worst = max(anomalies, key=lambda a: a["z"])
        tips.append({
            "title": "ارتفاع غير معتاد",
            "text": (f"استهلاك شهر {worst['month']} كان أعلى من المتوقع بحوالي {worst['pct_above']}%. "
                     "راجع الأجهزة التي تعمل باستمرار (سخان، تكييف، ثلاجة قديمة)، "
                     "وتأكد من قراءة العداد ومن عدم وجود تسريب كهربائي أو عطل في جهاز."),
            "priority": "high",
        })

    growth = stats.get("yearly_growth_pct")
    if growth is not None and growth > 5:
        tips.append({
            "title": "اتجاه تصاعدي",
            "text": f"الاستهلاك زاد {growth}% عن العام السابق. راجع الأجهزة الجديدة وعادات التشغيل، "
                    "وفكّر في أجهزة موفرة للطاقة.",
            "priority": "high",
        })

    peaks = set(stats.get("peak_calendar_months", []))
    next_months = {int(f["month"][5:7]) for f in forecast}
    if (peaks & SUMMER_MONTHS) or (next_months & SUMMER_MONTHS):
        tips.append({
            "title": "التكييف في الصيف",
            "text": "اضبط التكييف على 24–25°م، نظّف الفلاتر بانتظام، أغلق الأبواب والنوافذ، "
                    "واستخدم الستائر لتقليل حرارة الشمس. التكييف الإنفرتر أوفر على المدى الطويل.",
            "priority": "medium",
        })

    tips += [
        {"title": "الإضاءة", "text": "استبدل اللمبات القديمة بـ LED، وأطفئ الإضاءة في الغرف الفارغة.", "priority": "low"},
        {"title": "الوضع الاحتياطي", "text": "افصل الأجهزة (تلفزيون، شاحن، راوتر غير مستخدم) أو استخدم مشترك بمفتاح؛ "
                                           "الـ Standby يستهلك طوال اليوم.", "priority": "low"},
        {"title": "سخان المياه", "text": "اضبط حرارته على مستوى معتدل واستخدم مؤقّتًا (Timer) بدل تركه شغّالًا طوال اليوم.", "priority": "low"},
        {"title": "الأجهزة الكبيرة", "text": "شغّل الغسالة والغسّالة الأطباق بحمل كامل، وابتعد عن تشغيل عدة أجهزة ثقيلة معًا.", "priority": "low"},
    ]
    return tips
