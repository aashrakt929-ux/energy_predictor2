"""Flask API + قاعدة البيانات (MySQL في الإنتاج، SQLite افتراضيًا للتجربة المحلية)."""
import io
import os
import smtplib
import time
from datetime import date, datetime
from email.message import EmailMessage

import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import OperationalError

import ml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# مثال MySQL: mysql+pymysql://user:password@host:3306/energy
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///energy.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
app.json.ensure_ascii = False
db = SQLAlchemy(app)


# ---------------------------------------------------------------- النماذج
class Reading(db.Model):
    __tablename__ = "readings"
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False, unique=True, index=True)
    kwh = db.Column(db.Float, nullable=False)


class Alert(db.Model):
    __tablename__ = "alerts"
    id = db.Column(db.Integer, primary_key=True)
    month = db.Column(db.String(7), nullable=False, unique=True)
    kwh = db.Column(db.Float, nullable=False)
    expected = db.Column(db.Float, nullable=False)
    pct_above = db.Column(db.Float)
    message = db.Column(db.String(255), nullable=False)
    acknowledged = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self):
        return {
            "id": self.id, "month": self.month, "kwh": self.kwh, "expected": self.expected,
            "pct_above": self.pct_above, "message": self.message,
            "acknowledged": self.acknowledged, "created_at": self.created_at.isoformat(),
        }


def init_db(retries: int = 15):
    """ينتظر قاعدة البيانات (مفيد مع Docker/MySQL) ثم ينشئ الجداول."""
    for attempt in range(retries):
        try:
            with app.app_context():
                db.create_all()
            return
        except OperationalError:
            if attempt == retries - 1:
                raise
            time.sleep(2)


init_db()


# ---------------------------------------------------------------- أدوات مساعدة
COLUMN_ALIASES = {
    "date": {"date", "month", "التاريخ", "الشهر"},
    "kwh": {"kwh", "consumption", "usage", "الاستهلاك", "كيلوواط"},
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        for target, names in COLUMN_ALIASES.items():
            if key in names:
                rename[col] = target
    df = df.rename(columns=rename)
    if not {"date", "kwh"} <= set(df.columns):
        raise ValueError("الملف لازم يحتوي على عمودين: date و kwh")
    return df[["date", "kwh"]]


def save_readings(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["kwh"] = pd.to_numeric(df["kwh"], errors="coerce")
    total = len(df)
    df = df.dropna(subset=["date", "kwh"])
    df = df[df["kwh"] >= 0]
    skipped = total - len(df)

    added = updated = 0
    for d, kwh in zip(df["date"].dt.date, df["kwh"]):
        row = Reading.query.filter_by(date=d).first()
        if row:
            row.kwh = float(kwh)
            updated += 1
        else:
            db.session.add(Reading(date=d, kwh=float(kwh)))
            added += 1
    db.session.commit()
    return {"added": added, "updated": updated, "skipped": skipped}


def readings_df() -> pd.DataFrame:
    rows = Reading.query.order_by(Reading.date).all()
    return pd.DataFrame([{"date": r.date, "kwh": r.kwh} for r in rows], columns=["date", "kwh"])


def send_email_alert(alert: Alert):
    """إرسال بريد اختياري عند تفعيل متغيرات SMTP_HOST و ALERT_EMAIL_TO."""
    host, to = os.getenv("SMTP_HOST"), os.getenv("ALERT_EMAIL_TO")
    if not host or not to:
        return
    try:
        msg = EmailMessage()
        msg["Subject"] = f"تنبيه استهلاك كهرباء - {alert.month}"
        msg["From"] = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", "alerts@localhost"))
        msg["To"] = to
        msg.set_content(alert.message)
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=10) as s:
            s.starttls()
            if os.getenv("SMTP_USER"):
                s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD", ""))
            s.send_message(msg)
    except Exception as exc:  # التنبيه لا يجب أن يكسر التحليل
        app.logger.warning("email alert failed: %s", exc)


def sync_alerts(anomalies: list[dict]) -> list[str]:
    """ينشئ/يحدّث تنبيهات الشذوذ، ويرجّع الشهور الجديدة فقط."""
    current = {a["month"] for a in anomalies}
    new_months = []
    for a in anomalies:
        msg = (f"استهلاك {a['month']} = {a['kwh']} kWh، أعلى من المتوقع ({a['expected']} kWh) "
               f"بنسبة {a['pct_above']}%.")
        alert = Alert.query.filter_by(month=a["month"]).first()
        if alert is None:
            alert = Alert(month=a["month"], kwh=a["kwh"], expected=a["expected"],
                          pct_above=a["pct_above"], message=msg)
            db.session.add(alert)
            db.session.flush()
            new_months.append(a["month"])
            send_email_alert(alert)
        else:
            alert.kwh, alert.expected, alert.pct_above, alert.message = (
                a["kwh"], a["expected"], a["pct_above"], msg)
    for stale in Alert.query.filter_by(acknowledged=False).all():
        if stale.month not in current:
            db.session.delete(stale)
    db.session.commit()
    return new_months


# ---------------------------------------------------------------- المسارات
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify(error="اختر ملف CSV"), 400
    try:
        raw = pd.read_csv(io.BytesIO(file.read()))
        summary = save_readings(normalize_columns(raw))
    except Exception as exc:
        db.session.rollback()
        return jsonify(error=f"تعذّرت قراءة الملف: {exc}"), 400
    return jsonify(summary)


@app.post("/api/readings")
def add_reading():
    data = request.get_json(silent=True) or {}
    try:
        d = pd.to_datetime(data.get("date")).date()
        kwh = float(data.get("kwh"))
        if kwh < 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify(error="أدخل تاريخًا صحيحًا واستهلاكًا (kWh) رقمًا موجبًا"), 400
    summary = save_readings(pd.DataFrame([{"date": d, "kwh": kwh}]))
    return jsonify(summary), 201


@app.get("/api/readings")
def list_readings():
    rows = Reading.query.order_by(Reading.date).all()
    return jsonify([{"date": r.date.isoformat(), "kwh": r.kwh} for r in rows])


@app.delete("/api/readings")
def clear_readings():
    Reading.query.delete()
    Alert.query.delete()
    db.session.commit()
    return jsonify(ok=True)


@app.post("/api/sample")
def load_sample():
    df = pd.read_csv(os.path.join(BASE_DIR, "sample_data.csv"))
    return jsonify(save_readings(df))


@app.get("/api/analysis")
def analysis():
    try:
        horizon = max(1, min(int(request.args.get("horizon", 3)), 12))
    except ValueError:
        horizon = 3
    df = readings_df()
    if df.empty:
        return jsonify(error="لا توجد بيانات بعد. أضف قراءات أو حمّل بيانات تجريبية."), 400
    try:
        result = ml.analyze(df, horizon=horizon)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    new_months = sync_alerts(result["anomalies"])
    result["new_alert_months"] = new_months
    result["alerts"] = [a.to_dict() for a in Alert.query.order_by(Alert.month.desc()).all()]
    return jsonify(result)


@app.get("/api/alerts")
def list_alerts():
    return jsonify([a.to_dict() for a in Alert.query.order_by(Alert.month.desc()).all()])


@app.post("/api/alerts/<int:alert_id>/ack")
def ack_alert(alert_id: int):
    alert = db.get_or_404(Alert, alert_id)
    alert.acknowledged = True
    db.session.commit()
    return jsonify(alert.to_dict())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
