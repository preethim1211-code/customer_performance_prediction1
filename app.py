"""
app.py  –  CustPredict v4
--------------------------
ALL features:
  ✓ Self-registration / login (username or email)
  ✓ Per-user data isolation
  ✓ XGBoost + SHAP + LR + RF predictions
  ✓ CLV Score + gauge
  ✓ Prediction history (saved on every edit)
  ✓ Customer activity timeline
  ✓ Customer tags / labels
  ✓ Scatter plot (Spending vs Frequency, coloured by segment)
  ✓ Dashboard date-range filter + month-over-month comparison
  ✓ Batch status update (checkboxes)
  ✓ Customer notes history (timestamped)
  ✓ Search across all fields
  ✓ Dark / Light mode toggle
  ✓ Loading spinner on predict
  ✓ Profile photo upload
  ✓ Personal API key per user
  ✓ CSV export / bulk import (robust)
  ✓ PDF report (single + full dashboard)
  ✓ Forgot / reset password (token-based)
  ✓ AI Chatbot (calls Anthropic API)
  ✓ Data Quality Checker (single record + full CSV)
  ✓ Audit log + Admin panel
"""

import os, io, csv, json, re, secrets, hashlib
from datetime import datetime, date, timedelta
from functools import wraps

import joblib
import numpy as np
import shap

from flask import (Flask, render_template, request, redirect,
                   url_for, flash, jsonify, Response, abort,
                   session, send_from_directory)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user,
                         logout_user, login_required, current_user)
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import xgboost as xgb

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)
from reportlab.lib.units import cm

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.update(
    SECRET_KEY                     = 'cpps-v4-secret-2024',
    SQLALCHEMY_DATABASE_URI        = 'sqlite:///customers.db',
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    WTF_CSRF_ENABLED               = True,
    MAX_CONTENT_LENGTH             = 10 * 1024 * 1024,
    UPLOAD_FOLDER                  = os.path.join('static', 'uploads'),
    # Flask-Mail (configure with real SMTP to enable email features)
    MAIL_SERVER                    = os.environ.get('MAIL_SERVER', 'smtp.gmail.com'),
    MAIL_PORT                      = 587,
    MAIL_USE_TLS                   = True,
    MAIL_USERNAME                  = os.environ.get('MAIL_USERNAME', ''),
    MAIL_PASSWORD                  = os.environ.get('MAIL_PASSWORD', ''),
    MAIL_DEFAULT_SENDER            = os.environ.get('MAIL_USERNAME', 'noreply@custpredict.com'),
)

db            = SQLAlchemy(app)
csrf          = CSRFProtect(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

try:
    from flask_mail import Mail, Message
    mail = Mail(app)
    MAIL_ENABLED = bool(app.config['MAIL_USERNAME'])
except Exception:
    MAIL_ENABLED = False

ALLOWED_IMG = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# ── Load ML models ─────────────────────────────────────────────────────────────
M = 'models'
def load(n): return joblib.load(os.path.join(M, n))

def safe_load(name):
    try:
        print(f"Loading {name}...")
        return load(name)
    except Exception as e:
        print(f"Error loading {name}:", e)
        return None

lr_churn       = safe_load('lr_churn.pkl')
rf_churn       = safe_load('rf_churn.pkl')
xgb_churn      = safe_load('xgb_churn.pkl')
lr_value       = safe_load('lr_value.pkl')
rf_value       = safe_load('rf_value.pkl')
xgb_value      = safe_load('xgb_value.pkl')
scaler_churn   = safe_load('scaler_churn.pkl')
scaler_value   = safe_load('scaler_value.pkl')
scaler_kmeans  = safe_load('scaler_kmeans.pkl')
kmeans         = safe_load('kmeans.pkl')
le_gender      = safe_load('le_gender.pkl')
le_value       = safe_load('le_value.pkl')

shap_explainer = None  # disable SHAP

with open(os.path.join(M, 'meta.json')) as f:
    META = json.load(f)

SEGMENT_NAMES = {0: 'Bronze', 1: 'Silver', 2: 'Gold'}
FEATURE_NAMES = META['feature_names']

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE MODELS
# ══════════════════════════════════════════════════════════════════════════════

class User(UserMixin, db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    full_name     = db.Column(db.String(120), default='')
    password_hash = db.Column(db.String(200), nullable=False)
    role          = db.Column(db.String(20),  default='user')
    api_key       = db.Column(db.String(64),  unique=True)
    avatar        = db.Column(db.String(200), default='')
    theme         = db.Column(db.String(10),  default='dark')
    reset_token   = db.Column(db.String(100), nullable=True)
    reset_expiry  = db.Column(db.DateTime,    nullable=True)
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    customers     = db.relationship('Customer', backref='owner', lazy=True,
                                    cascade='all, delete-orphan')

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)
    def generate_api_key(self):
        self.api_key = secrets.token_hex(32)


class Customer(db.Model):
    id                 = db.Column(db.Integer,  primary_key=True)
    user_id            = db.Column(db.Integer,  db.ForeignKey('user.id'), nullable=False)
    name               = db.Column(db.String(120), nullable=False)
    age                = db.Column(db.Integer,  nullable=False)
    gender             = db.Column(db.String(10), nullable=False)
    purchase_freq      = db.Column(db.Integer,  nullable=False)
    total_spending     = db.Column(db.Float,    nullable=False)
    last_purchase_date = db.Column(db.Date,     nullable=False)
    status             = db.Column(db.String(20), default='Active')
    tags               = db.Column(db.String(200), default='')   # comma-separated
    # Predictions
    churn_lr           = db.Column(db.String(5))
    churn_rf           = db.Column(db.String(5))
    churn_xgb          = db.Column(db.String(5))
    churn_prob_lr      = db.Column(db.Float, default=0)
    churn_prob_rf      = db.Column(db.Float, default=0)
    churn_prob_xgb     = db.Column(db.Float, default=0)
    value_lr           = db.Column(db.String(10))
    value_rf           = db.Column(db.String(10))
    value_xgb          = db.Column(db.String(10))
    segment            = db.Column(db.String(10))
    clv_score          = db.Column(db.Float, default=0)
    shap_json          = db.Column(db.Text,  default='{}')
    created_at         = db.Column(db.DateTime, default=datetime.utcnow)
    # Relationships
    notes_history      = db.relationship('CustomerNote', backref='customer', lazy=True,
                                          cascade='all, delete-orphan',
                                          order_by='CustomerNote.created_at.desc()')
    pred_history       = db.relationship('PredictionHistory', backref='customer', lazy=True,
                                          cascade='all, delete-orphan',
                                          order_by='PredictionHistory.created_at.desc()')
    timeline_events    = db.relationship('TimelineEvent', backref='customer', lazy=True,
                                          cascade='all, delete-orphan',
                                          order_by='TimelineEvent.created_at.desc()')


class CustomerNote(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    note        = db.Column(db.Text,    nullable=False)
    author      = db.Column(db.String(80))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionHistory(db.Model):
    id            = db.Column(db.Integer, primary_key=True)
    customer_id   = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    churn_prob_rf = db.Column(db.Float)
    churn_rf      = db.Column(db.String(5))
    value_rf      = db.Column(db.String(10))
    segment       = db.Column(db.String(10))
    clv_score     = db.Column(db.Float)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


class TimelineEvent(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.Integer, db.ForeignKey('customer.id'), nullable=False)
    event_type  = db.Column(db.String(30))   # created|edited|status_change|note_added
    description = db.Column(db.String(200))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    user      = db.Column(db.String(80))
    action    = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(uid):
    return db.session.get(User, int(uid))

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            abort(403)
        return f(*args, **kwargs)
    return decorated

def api_key_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        user = User.query.filter_by(api_key=key).first() if key else None
        if not user:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        from flask_login import login_user
        login_user(user)
        return f(*args, **kwargs)
    return decorated

def log_action(action):
    db.session.add(AuditLog(user=current_user.username, action=action))
    db.session.commit()

def my_customers():
    return Customer.query.filter_by(user_id=current_user.id)

def add_timeline(customer, event_type, description):
    db.session.add(TimelineEvent(
        customer_id=customer.id,
        event_type=event_type,
        description=description
    ))

def calc_clv(purchase_freq, total_spending, days_since_last):
    """
    Customer Lifetime Value score (0-100).
    Formula: weighted combination of recency, frequency, monetary value.
    """
    recency_score   = max(0, 1 - days_since_last / 365)
    frequency_score = min(purchase_freq / 50, 1)
    monetary_score  = min(total_spending / 5000, 1)
    clv = (recency_score * 0.3 + frequency_score * 0.35 + monetary_score * 0.35) * 100
    return round(clv, 1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMG

def validate_registration(username, email, password, confirm):
    errors = []
    if len(username) < 3:
        errors.append('Username must be at least 3 characters.')
    if not re.match(r'^[A-Za-z0-9_]+$', username):
        errors.append('Username: letters, numbers and underscores only.')
    if User.query.filter_by(username=username).first():
        errors.append('That username is already taken.')
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        errors.append('Please enter a valid email address.')
    if User.query.filter_by(email=email).first():
        errors.append('An account with that email already exists.')
    if len(password) < 6:
        errors.append('Password must be at least 6 characters.')
    if password != confirm:
        errors.append('Passwords do not match.')
    return errors

# ══════════════════════════════════════════════════════════════════════════════
#  DATA QUALITY CHECKER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Statistical thresholds learned from training data
DQ_THRESHOLDS = {
    'age':           {'min': 18,   'max': 100,  'typical_min': 20,  'typical_max': 70},
    'purchase_freq': {'min': 1,    'max': 365,  'typical_min': 1,   'typical_max': 50},
    'total_spending':{'min': 0,    'max': 1e7,  'typical_min': 50,  'typical_max': 5000},
    'days_since':    {'min': 0,    'max': 3650, 'typical_min': 1,   'typical_max': 365},
}

def check_single_record(name, age, gender, purchase_freq, total_spending,
                         last_purchase_date, row_num=None):
    """
    Validate one customer record. Returns a dict with:
      - issues   : list of {field, severity, message} dicts
      - score    : int 0-100 (100 = perfect)
      - passed   : bool (True if no CRITICAL issues)
      - summary  : short human-readable string
    Severity levels: 'critical' | 'warning' | 'info'
    """
    issues  = []
    prefix  = f"Row {row_num}: " if row_num else ""

    # ── Name checks ───────────────────────────────────────────────────────────
    if not name or not name.strip():
        issues.append({'field':'name','severity':'critical',
                       'message': f'{prefix}Name is empty.'})
    elif len(name.strip()) < 2:
        issues.append({'field':'name','severity':'warning',
                       'message': f'{prefix}Name "{name}" is very short (< 2 chars).'})
    elif re.search(r'\d', name):
        issues.append({'field':'name','severity':'warning',
                       'message': f'{prefix}Name "{name}" contains numbers — is this correct?'})

    # ── Age checks ────────────────────────────────────────────────────────────
    try:
        age = int(float(str(age)))
        if age < 0:
            issues.append({'field':'age','severity':'critical',
                           'message': f'{prefix}Age {age} is negative.'})
        elif age < DQ_THRESHOLDS['age']['min']:
            issues.append({'field':'age','severity':'critical',
                           'message': f'{prefix}Age {age} is below minimum allowed (18).'})
        elif age > DQ_THRESHOLDS['age']['max']:
            issues.append({'field':'age','severity':'critical',
                           'message': f'{prefix}Age {age} exceeds maximum (100).'})
        elif age < DQ_THRESHOLDS['age']['typical_min']:
            issues.append({'field':'age','severity':'warning',
                           'message': f'{prefix}Age {age} is unusually young — verify.'})
        elif age > DQ_THRESHOLDS['age']['typical_max']:
            issues.append({'field':'age','severity':'warning',
                           'message': f'{prefix}Age {age} is unusually high — verify.'})
    except (ValueError, TypeError):
        issues.append({'field':'age','severity':'critical',
                       'message': f'{prefix}Age "{age}" is not a valid number.'})

    # ── Gender checks ─────────────────────────────────────────────────────────
    if str(gender).strip().capitalize() not in ('Male', 'Female'):
        issues.append({'field':'gender','severity':'critical',
                       'message': f'{prefix}Gender "{gender}" must be "Male" or "Female".'})

    # ── Purchase frequency checks ─────────────────────────────────────────────
    try:
        pfreq = int(float(str(purchase_freq)))
        if pfreq <= 0:
            issues.append({'field':'purchase_freq','severity':'critical',
                           'message': f'{prefix}Purchase frequency {pfreq} must be > 0.'})
        elif pfreq > DQ_THRESHOLDS['purchase_freq']['max']:
            issues.append({'field':'purchase_freq','severity':'critical',
                           'message': f'{prefix}Purchase frequency {pfreq} exceeds 365/year — impossible.'})
        elif pfreq > DQ_THRESHOLDS['purchase_freq']['typical_max']:
            issues.append({'field':'purchase_freq','severity':'warning',
                           'message': f'{prefix}Purchase frequency {pfreq}/yr is very high — verify.'})
    except (ValueError, TypeError):
        issues.append({'field':'purchase_freq','severity':'critical',
                       'message': f'{prefix}Purchase frequency "{purchase_freq}" is not a valid number.'})

    # ── Total spending checks ─────────────────────────────────────────────────
    try:
        spend = float(str(total_spending).replace(',',''))
        if spend < 0:
            issues.append({'field':'total_spending','severity':'critical',
                           'message': f'{prefix}Spending ${spend:.2f} is negative — impossible.'})
        elif spend == 0:
            issues.append({'field':'total_spending','severity':'warning',
                           'message': f'{prefix}Spending is $0.00 — is this customer inactive?'})
        elif spend < DQ_THRESHOLDS['total_spending']['typical_min']:
            issues.append({'field':'total_spending','severity':'info',
                           'message': f'{prefix}Spending ${spend:.2f} is very low (< $50) — possible data entry issue.'})
        elif spend > DQ_THRESHOLDS['total_spending']['typical_max']:
            issues.append({'field':'total_spending','severity':'warning',
                           'message': f'{prefix}Spending ${spend:,.0f} is above typical range — verify.'})
        elif spend > DQ_THRESHOLDS['total_spending']['max']:
            issues.append({'field':'total_spending','severity':'critical',
                           'message': f'{prefix}Spending ${spend:,.0f} is extremely large — likely a data error.'})
    except (ValueError, TypeError):
        issues.append({'field':'total_spending','severity':'critical',
                       'message': f'{prefix}Spending "{total_spending}" is not a valid number.'})

    # ── Last purchase date checks ─────────────────────────────────────────────
    try:
        if isinstance(last_purchase_date, str):
            lp = None
            for fmt in ('%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%d-%m-%Y'):
                try: lp = datetime.strptime(last_purchase_date.strip(), fmt).date(); break
                except: continue
            if lp is None:
                raise ValueError(f'unrecognised date format: {last_purchase_date}')
        else:
            lp = last_purchase_date

        days_since = (date.today() - lp).days
        if lp > date.today():
            issues.append({'field':'last_purchase_date','severity':'critical',
                           'message': f'{prefix}Last purchase date {lp} is in the future.'})
        elif days_since < 0:
            issues.append({'field':'last_purchase_date','severity':'critical',
                           'message': f'{prefix}Date is in the future — impossible.'})
        elif days_since > DQ_THRESHOLDS['days_since']['max']:
            issues.append({'field':'last_purchase_date','severity':'warning',
                           'message': f'{prefix}Last purchase was {days_since} days ago (>10 years) — verify.'})
        elif days_since > DQ_THRESHOLDS['days_since']['typical_max']:
            issues.append({'field':'last_purchase_date','severity':'info',
                           'message': f'{prefix}Last purchase was {days_since} days ago — customer may be inactive.'})
    except Exception as e:
        issues.append({'field':'last_purchase_date','severity':'critical',
                       'message': f'{prefix}Invalid date: {e}'})

    # ── Cross-field logic checks ──────────────────────────────────────────────
    try:
        pfreq = int(float(str(purchase_freq)))
        spend = float(str(total_spending).replace(',',''))
        if pfreq > 20 and spend < 100:
            issues.append({'field':'cross','severity':'warning',
                           'message': f'{prefix}High frequency ({pfreq}/yr) but very low spending (${spend:.0f}) — unusual pattern.'})
        if pfreq < 2 and spend > 3000:
            issues.append({'field':'cross','severity':'info',
                           'message': f'{prefix}Very low frequency ({pfreq}/yr) but high spending (${spend:,.0f}) — possible big-ticket buyer.'})
    except: pass

    # ── Score calculation ─────────────────────────────────────────────────────
    critical_count = sum(1 for i in issues if i['severity'] == 'critical')
    warning_count  = sum(1 for i in issues if i['severity'] == 'warning')
    info_count     = sum(1 for i in issues if i['severity'] == 'info')

    score = max(0, 100 - (critical_count * 30) - (warning_count * 10) - (info_count * 3))
    passed = critical_count == 0

    if score == 100:
        summary = 'Perfect — no issues found'
    elif score >= 80:
        summary = f'Good — {warning_count} warning(s), {info_count} note(s)'
    elif score >= 50:
        summary = f'Fair — {critical_count} critical, {warning_count} warning(s)'
    else:
        summary = f'Poor — {critical_count} critical issue(s) must be fixed'

    return {
        'issues':   issues,
        'score':    score,
        'passed':   passed,
        'summary':  summary,
        'counts':   {'critical': critical_count, 'warning': warning_count, 'info': info_count},
    }


def check_csv_quality(content_str):
    """
    Run quality checks on an entire CSV string.
    Returns overall report + per-row results.
    """
    stream = io.StringIO(content_str)
    reader = csv.DictReader(stream)

    if not reader.fieldnames:
        return {'error': 'File is empty or has no headers.'}

    # Normalise column names
    norm = {k.strip().lower().replace(' ','_').replace('-','_'): k
            for k in reader.fieldnames}
    REQUIRED = {'name','age','gender','purchase_freq','total_spending','last_purchase_date'}
    missing  = REQUIRED - set(norm.keys())
    if missing:
        return {'error': f'Missing columns: {", ".join(sorted(missing))}'}

    def get(row, k):
        return row.get(norm.get(k,''), '').strip()

    rows_checked   = []
    total_score    = 0
    critical_rows  = 0
    all_issues     = []

    for i, row in enumerate(reader, start=2):
        name     = get(row, 'name')
        age      = get(row, 'age')
        gender   = get(row, 'gender')
        pfreq    = get(row, 'purchase_freq')
        spending = get(row, 'total_spending')
        lpdate   = get(row, 'last_purchase_date')

        result = check_single_record(name, age, gender, pfreq, spending, lpdate, row_num=i)
        rows_checked.append({
            'row':     i,
            'name':    name or f'Row {i}',
            'score':   result['score'],
            'passed':  result['passed'],
            'summary': result['summary'],
            'issues':  result['issues'],
            'counts':  result['counts'],
        })
        total_score += result['score']
        if not result['passed']:
            critical_rows += 1
        all_issues.extend(result['issues'])

    total_rows    = len(rows_checked)
    avg_score     = round(total_score / total_rows, 1) if total_rows else 0
    clean_rows    = sum(1 for r in rows_checked if r['score'] == 100)
    warning_rows  = sum(1 for r in rows_checked if 0 < r['counts']['warning'] and r['passed'])

    # Field-level issue summary
    field_counts = {}
    for iss in all_issues:
        f = iss['field']
        field_counts[f] = field_counts.get(f, 0) + 1

    return {
        'total_rows':    total_rows,
        'clean_rows':    clean_rows,
        'critical_rows': critical_rows,
        'warning_rows':  warning_rows,
        'avg_score':     avg_score,
        'field_counts':  field_counts,
        'rows':          rows_checked,
        'can_import':    critical_rows == 0,
    }


# ── ML helpers ─────────────────────────────────────────────────────────────────
def build_features(age, gender, purchase_freq, total_spending, last_purchase_date):
    gender_enc      = le_gender.transform([gender])[0]
    days_since_last = (date.today() - last_purchase_date).days
    return np.array([[age, gender_enc, purchase_freq, total_spending, days_since_last]]), days_since_last

def predict_all(X_raw):
    # --- Safe transformations ---
    X_churn = scaler_churn.transform(X_raw) if scaler_churn else X_raw
    X_val   = scaler_value.transform(X_raw) if scaler_value else X_raw
    X_km    = scaler_kmeans.transform(X_raw) if scaler_kmeans else X_raw

    # --- Churn Predictions ---
    c_lr  = 'Yes' if lr_churn and lr_churn.predict(X_churn)[0] == 1 else 'No'
    c_rf  = 'Yes' if rf_churn and rf_churn.predict(X_raw)[0] == 1 else 'No'
    c_xgb = 'Yes' if xgb_churn and xgb_churn.predict(X_raw)[0] == 1 else 'No'

    # --- Probabilities ---
    p_lr  = round(float(lr_churn.predict_proba(X_churn)[0][1]) * 100, 1) if lr_churn else 0
    p_rf  = round(float(rf_churn.predict_proba(X_raw)[0][1]) * 100, 1) if rf_churn else 0
    p_xgb = round(float(xgb_churn.predict_proba(X_raw)[0][1]) * 100, 1) if xgb_churn else 0

    # --- Value Prediction ---
    v_lr  = le_value.inverse_transform(lr_value.predict(X_val))[0] if lr_value and le_value else "N/A"
    v_rf  = le_value.inverse_transform(rf_value.predict(X_raw))[0] if rf_value and le_value else "N/A"
    v_xgb = le_value.inverse_transform(xgb_value.predict(X_raw))[0] if xgb_value and le_value else "N/A"

    # --- Segmentation ---
    if kmeans:
        seg_id = kmeans.predict(X_km)[0]
        segment = SEGMENT_NAMES.get(int(seg_id), 'Unknown')
    else:
        segment = "Unavailable"

    # --- SHAP (safe) ---
    shap_dict = {}

    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(X_raw)

            if isinstance(sv, list):
                sv_churn = np.array(sv[1]).flatten()
            else:
                sv_arr = np.array(sv)
                if sv_arr.ndim == 3:
                    sv_churn = sv_arr[0, :, 1]
                elif sv_arr.ndim == 2:
                    sv_churn = sv_arr[0]
                else:
                    sv_churn = sv_arr.flatten()

            shap_dict = {
                FEATURE_NAMES[i]: round(float(sv_churn[i]), 4)
                for i in range(len(FEATURE_NAMES))
            }

        except Exception as e:
            print("SHAP error:", e)

    # --- Final return ---
    return {
        'churn_lr': c_lr,   'churn_prob_lr':  p_lr,
        'churn_rf': c_rf,   'churn_prob_rf':  p_rf,
        'churn_xgb': c_xgb, 'churn_prob_xgb': p_xgb,
        'value_lr': v_lr,   'value_rf': v_rf, 'value_xgb': v_xgb,
        'segment':  segment, 'shap': shap_dict,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return redirect(url_for('dashboard') if current_user.is_authenticated else url_for('register'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username  = request.form.get('username','').strip()
        email     = request.form.get('email','').strip().lower()
        full_name = request.form.get('full_name','').strip()
        password  = request.form.get('password','')
        confirm   = request.form.get('confirm_password','')

        errors = validate_registration(username, email, password, confirm)
        if errors:
            for e in errors: flash(e, 'danger')
            return render_template('register.html', username=username, email=email, full_name=full_name)

        role = 'admin' if User.query.count() == 0 else 'user'
        u = User(username=username, email=email, full_name=full_name, role=role)
        u.set_password(password)
        u.generate_api_key()
        db.session.add(u)
        db.session.commit()
        login_user(u)
        db.session.add(AuditLog(user=username, action='Registered'))
        db.session.commit()
        flash(f'Welcome, {username}! Your account has been created.', 'success')
        return redirect(url_for('dashboard'))
    return render_template('register.html', username='', email='', full_name='')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        identifier = request.form.get('identifier','').strip()
        password   = request.form.get('password','')
        u = (User.query.filter_by(username=identifier).first() or
             User.query.filter_by(email=identifier).first())
        if u and u.check_password(password):
            login_user(u)
            db.session.add(AuditLog(user=u.username, action='Logged in'))
            db.session.commit()
            return redirect(url_for('dashboard'))
        flash('Invalid username/email or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    log_action('Logged out')
    logout_user()
    return redirect(url_for('login'))

# ── Forgot / Reset password ────────────────────────────────────────────────────
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        u = User.query.filter_by(email=email).first()
        if u:
            token = secrets.token_urlsafe(32)
            u.reset_token  = token
            u.reset_expiry = datetime.utcnow() + timedelta(hours=1)
            db.session.commit()
            reset_link = url_for('reset_password', token=token, _external=True)
            if MAIL_ENABLED:
                try:
                    msg = Message('CustPredict — Password Reset',
                                  recipients=[email])
                    msg.body = f"Click to reset your password (valid 1 hour):\n{reset_link}"
                    mail.send(msg)
                    flash('Password reset link sent to your email.', 'success')
                except Exception:
                    flash(f'Email not configured. Reset link: {reset_link}', 'warning')
            else:
                flash(f'Email not configured. Your reset link: {reset_link}', 'info')
        else:
            flash('If that email exists, a reset link was sent.', 'info')
    return render_template('forgot_password.html')

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    u = User.query.filter_by(reset_token=token).first()
    if not u or not u.reset_expiry or u.reset_expiry < datetime.utcnow():
        flash('Reset link is invalid or expired.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_pw  = request.form.get('password','')
        confirm = request.form.get('confirm','')
        if len(new_pw) < 6:
            flash('Password must be at least 6 characters.', 'danger')
        elif new_pw != confirm:
            flash('Passwords do not match.', 'danger')
        else:
            u.set_password(new_pw)
            u.reset_token  = None
            u.reset_expiry = None
            db.session.commit()
            flash('Password reset! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template('reset_password.html', token=token)

# ── Theme toggle ───────────────────────────────────────────────────────────────
@app.route('/toggle-theme', methods=['POST'])
@login_required
def toggle_theme():
    current_user.theme = 'light' if current_user.theme == 'dark' else 'dark'
    db.session.commit()
    return jsonify({'theme': current_user.theme})

# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/dashboard')
@login_required
def dashboard():
    # Date range filter
    date_from = request.args.get('date_from','')
    date_to   = request.args.get('date_to','')

    query = my_customers()
    if date_from:
        try: query = query.filter(Customer.created_at >= datetime.strptime(date_from,'%Y-%m-%d'))
        except: pass
    if date_to:
        try: query = query.filter(Customer.created_at <= datetime.strptime(date_to,'%Y-%m-%d') + timedelta(days=1))
        except: pass

    customers = query.order_by(Customer.created_at.desc()).all()
    total     = len(customers)

    churn_yes = sum(1 for c in customers if c.churn_rf=='Yes')
    churn_no  = total - churn_yes
    value_counts  = {'High':0,'Medium':0,'Low':0}
    seg_counts    = {'Gold':0,'Silver':0,'Bronze':0}
    status_counts = {'Active':0,'At Risk':0,'Churned':0,'Retained':0}
    for c in customers:
        if c.value_rf in value_counts:  value_counts[c.value_rf]  += 1
        if c.segment  in seg_counts:    seg_counts[c.segment]     += 1
        if c.status   in status_counts: status_counts[c.status]   += 1

    # Monthly trend
    from collections import defaultdict
    monthly = defaultdict(int)
    for c in customers:
        monthly[c.created_at.strftime('%b %Y')] += 1
    month_labels = list(monthly.keys())[-6:]
    month_data   = [monthly[k] for k in month_labels]

    # Month-over-month comparison
    now      = datetime.utcnow()
    mo_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    lm_start = (mo_start - timedelta(days=1)).replace(day=1)
    this_month = my_customers().filter(Customer.created_at >= mo_start).count()
    last_month = my_customers().filter(Customer.created_at >= lm_start,
                                        Customer.created_at < mo_start).count()
    mom_change = this_month - last_month

    avg_spend  = round(sum(c.total_spending for c in customers)/total, 2) if total else 0
    avg_clv    = round(sum(c.clv_score for c in customers)/total, 1) if total else 0

    # Scatter data for segment plot
    scatter = [{'x': c.purchase_freq, 'y': c.total_spending,
                'segment': c.segment, 'name': c.name} for c in customers]

    stats = {'total':total,'churn_yes':churn_yes,'avg_spend':avg_spend,
             'high_value':value_counts['High'],'avg_clv':avg_clv,
             'this_month':this_month,'last_month':last_month,'mom_change':mom_change}
    chart = {
        'churn':[churn_yes,churn_no], 'value':list(value_counts.values()),
        'segment':list(seg_counts.values()), 'status':list(status_counts.values()),
        'monthly_labels':month_labels, 'monthly_data':month_data,
        'scatter': scatter,
    }
    return render_template('dashboard.html',
                           customers=customers[:10], stats=stats,
                           chart_data=json.dumps(chart),
                           date_from=date_from, date_to=date_to)

@app.route('/upload', methods=['POST'])
def upload():
    import pandas as pd

    try:
        file = request.files.get('file')

        if not file or file.filename == '':
            return "No file uploaded", 400

        print("📁 File received:", file.filename)

        # --- Read CSV ---
        try:
            df = pd.read_csv(file)
        except Exception:
            df = pd.read_csv(file, encoding='latin1')

        print("✅ CSV Loaded")
        print(df.head())

        # --- Clean Data ---
        # Drop unwanted columns
        df = df.drop(columns=['name', 'last_purchase_date'], errors='ignore')

        # Handle missing columns
        required_cols = FEATURE_NAMES
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            return f"Missing columns: {missing}", 400

        # Encode gender safely
        if 'gender' in df.columns:
            df['gender'] = df['gender'].map({
                'Male': 1,
                'Female': 0,
                'male': 1,
                'female': 0
            })

        # Fill missing values
        df = df.fillna(0)

        # Ensure correct order
        df = df[FEATURE_NAMES]

        print("✅ Processed Data")
        print(df.head())

        # --- Predictions ---
        results = []

        for _, row in df.iterrows():
            X_raw = row.values.reshape(1, -1)
            result = predict_all(X_raw)
            results.append(result)

        print("✅ Predictions done")

        return {
            "status": "success",
            "predictions": results
        }

    except Exception as e:
        print("❌ UPLOAD ERROR:", e)
        return f"Internal Server Error: {str(e)}", 500                          

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            name     = request.form['name'].strip()
            age      = int(request.form['age'])
            gender   = request.form['gender']
            pfreq    = int(request.form['purchase_freq'])
            spending = float(request.form['total_spending'])
            lpdate   = datetime.strptime(request.form['last_purchase_date'],'%Y-%m-%d').date()
            tags     = request.form.get('tags','').strip()
            note_txt = request.form.get('note','').strip()

            X_raw, days_since = build_features(age, gender, pfreq, spending, lpdate)
            result = predict_all(X_raw)
            result['name'] = name

            clv    = calc_clv(pfreq, spending, days_since)
            status = 'At Risk' if result['churn_rf']=='Yes' else 'Active'

            c = Customer(
                user_id=current_user.id, name=name, age=age, gender=gender,
                purchase_freq=pfreq, total_spending=spending,
                last_purchase_date=lpdate, status=status, tags=tags,
                clv_score=clv,
                churn_lr=result['churn_lr'],   churn_prob_lr=result['churn_prob_lr'],
                churn_rf=result['churn_rf'],   churn_prob_rf=result['churn_prob_rf'],
                churn_xgb=result['churn_xgb'], churn_prob_xgb=result['churn_prob_xgb'],
                value_lr=result['value_lr'],   value_rf=result['value_rf'],
                value_xgb=result['value_xgb'], segment=result['segment'],
                shap_json=json.dumps(result['shap']),
            )
            db.session.add(c)
            db.session.flush()

            # Save initial prediction history
            db.session.add(PredictionHistory(
                customer_id=c.id, churn_prob_rf=result['churn_prob_rf'],
                churn_rf=result['churn_rf'], value_rf=result['value_rf'],
                segment=result['segment'], clv_score=clv,
            ))
            # Timeline event
            add_timeline(c, 'created', f'Customer added with churn risk {result["churn_prob_rf"]}%')
            # Note
            if note_txt:
                db.session.add(CustomerNote(customer_id=c.id, note=note_txt, author=current_user.username))
                add_timeline(c, 'note_added', f'Note: {note_txt[:60]}')

            db.session.commit()
            result['id']  = c.id
            result['clv'] = clv

            if result['churn_prob_rf'] > 70:
                flash(f"⚠️ High churn risk for {name} ({result['churn_prob_rf']}%)!", 'warning')
            log_action(f"Added customer: {name} (ID {c.id})")

        except Exception as e:
            flash(f'Prediction error: {e}', 'danger')
    return render_template('predict.html', result=result)

# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMERS
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/customers')
@login_required
def customers():
    q       = request.args.get('q','').strip()
    churn_f = request.args.get('churn','')
    value_f = request.args.get('value','')
    seg_f   = request.args.get('segment','')
    tag_f   = request.args.get('tag','')
    sort    = request.args.get('sort','date')
    page    = request.args.get('page',1,type=int)

    query = my_customers()
    if q:
        query = query.filter(
            db.or_(
                Customer.name.ilike(f'%{q}%'),
                Customer.gender.ilike(f'%{q}%'),
                Customer.segment.ilike(f'%{q}%'),
                Customer.status.ilike(f'%{q}%'),
                Customer.tags.ilike(f'%{q}%'),
            )
        )
    if churn_f: query = query.filter(Customer.churn_rf==churn_f)
    if value_f: query = query.filter(Customer.value_rf==value_f)
    if seg_f:   query = query.filter(Customer.segment==seg_f)
    if tag_f:   query = query.filter(Customer.tags.ilike(f'%{tag_f}%'))

    if sort=='spend': query = query.order_by(Customer.total_spending.desc())
    elif sort=='age': query = query.order_by(Customer.age)
    elif sort=='clv': query = query.order_by(Customer.clv_score.desc())
    else:             query = query.order_by(Customer.created_at.desc())

    total = query.count()
    items = query.offset((page-1)*20).limit(20).all()
    pages = max(1,(total+19)//20)

    # All unique tags for filter dropdown
    all_tags = set()
    for c in my_customers().all():
        for t in (c.tags or '').split(','):
            t = t.strip()
            if t: all_tags.add(t)

    return render_template('customers.html',
                           customers=items, page=page, pages=pages, total=total,
                           q=q, churn_f=churn_f, value_f=value_f,
                           seg_f=seg_f, sort=sort, tag_f=tag_f,
                           all_tags=sorted(all_tags))

# ── Batch status update ────────────────────────────────────────────────────────
@app.route('/customers/batch-update', methods=['POST'])
@login_required
def batch_update():
    ids    = request.form.getlist('selected_ids')
    status = request.form.get('batch_status','')
    if ids and status:
        for cid in ids:
            c = my_customers().filter_by(id=int(cid)).first()
            if c:
                old_status = c.status
                c.status   = status
                add_timeline(c, 'status_change', f'Status: {old_status} → {status} (batch)')
        db.session.commit()
        flash(f'Updated {len(ids)} customers to "{status}".', 'success')
        log_action(f'Batch update {len(ids)} customers → {status}')
    return redirect(url_for('customers'))

# ── Customer detail ────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>')
@login_required
def customer_detail(cid):
    c = my_customers().filter_by(id=cid).first_or_404()
    shap_data = json.loads(c.shap_json or '{}')
    # Prediction history for chart
    ph = [{'date': p.created_at.strftime('%d %b'), 'prob': p.churn_prob_rf,
            'clv': p.clv_score} for p in c.pred_history]
    return render_template('customer_detail.html', c=c, shap_data=shap_data, pred_history=ph)

# ── Add note ───────────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>/note', methods=['POST'])
@login_required
def add_note(cid):
    c    = my_customers().filter_by(id=cid).first_or_404()
    note = request.form.get('note','').strip()
    if note:
        db.session.add(CustomerNote(customer_id=c.id, note=note, author=current_user.username))
        add_timeline(c, 'note_added', f'Note: {note[:80]}')
        db.session.commit()
        flash('Note added.', 'success')
    return redirect(url_for('customer_detail', cid=cid))

# ── Edit ───────────────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>/edit', methods=['GET','POST'])
@login_required
def customer_edit(cid):
    c = my_customers().filter_by(id=cid).first_or_404()
    if request.method == 'POST':
        try:
            old_status = c.status
            c.name               = request.form['name'].strip()
            c.age                = int(request.form['age'])
            c.gender             = request.form['gender']
            c.purchase_freq      = int(request.form['purchase_freq'])
            c.total_spending     = float(request.form['total_spending'])
            c.last_purchase_date = datetime.strptime(request.form['last_purchase_date'],'%Y-%m-%d').date()
            c.status             = request.form.get('status', c.status)
            c.tags               = request.form.get('tags','').strip()

            X_raw, days_since = build_features(c.age, c.gender, c.purchase_freq,
                                               c.total_spending, c.last_purchase_date)
            res = predict_all(X_raw)
            clv = calc_clv(c.purchase_freq, c.total_spending, days_since)

            # Save prediction history snapshot
            db.session.add(PredictionHistory(
                customer_id=c.id, churn_prob_rf=res['churn_prob_rf'],
                churn_rf=res['churn_rf'], value_rf=res['value_rf'],
                segment=res['segment'], clv_score=clv,
            ))

            c.churn_lr=res['churn_lr'];  c.churn_prob_lr=res['churn_prob_lr']
            c.churn_rf=res['churn_rf'];  c.churn_prob_rf=res['churn_prob_rf']
            c.churn_xgb=res['churn_xgb']; c.churn_prob_xgb=res['churn_prob_xgb']
            c.value_lr=res['value_lr'];  c.value_rf=res['value_rf']
            c.value_xgb=res['value_xgb']; c.segment=res['segment']
            c.shap_json=json.dumps(res['shap']); c.clv_score=clv

            add_timeline(c, 'edited', f'Profile updated, churn risk now {res["churn_prob_rf"]}%')
            if old_status != c.status:
                add_timeline(c, 'status_change', f'Status: {old_status} → {c.status}')

            db.session.commit()
            log_action(f"Edited customer ID {cid}")
            flash('Customer updated and re-predicted.', 'success')
            return redirect(url_for('customer_detail', cid=cid))
        except Exception as e:
            flash(f'Update error: {e}', 'danger')
    return render_template('customer_edit.html', c=c)

# ── Delete ─────────────────────────────────────────────────────────────────────
@app.route('/customer/<int:cid>/delete', methods=['POST'])
@login_required
def customer_delete(cid):
    c = my_customers().filter_by(id=cid).first_or_404()
    name = c.name
    db.session.delete(c)
    db.session.commit()
    log_action(f"Deleted customer: {name}")
    flash(f'Customer "{name}" deleted.', 'success')
    return redirect(url_for('customers'))

# ══════════════════════════════════════════════════════════════════════════════
#  CSV EXPORT / IMPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/export/csv')
@login_required
def export_csv():
    custs = my_customers().order_by(Customer.created_at.desc()).all()
    si = io.StringIO()
    w  = csv.writer(si)
    w.writerow(['ID','Name','Age','Gender','Freq','Spending','Last Purchase',
                'Status','Tags','CLV Score','Churn RF','Value RF','Segment'])
    for c in custs:
        w.writerow([c.id,c.name,c.age,c.gender,c.purchase_freq,c.total_spending,
                    c.last_purchase_date,c.status,c.tags,c.clv_score,
                    c.churn_rf,c.value_rf,c.segment])
    log_action('Exported CSV')
    return Response(si.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition':
                             f'attachment; filename={current_user.username}_customers.csv'})

@app.route('/import/csv', methods=['GET','POST'])
@login_required
def import_csv():
    results = []
    if request.method == 'POST':
        f = request.files.get('csv_file')
        if not f or not f.filename.lower().endswith('.csv'):
            flash('Please upload a valid .csv file.', 'danger')
            return redirect(url_for('import_csv'))

        raw_bytes = f.stream.read()
        content = None
        for enc in ('utf-8-sig','utf-8','latin-1','cp1252'):
            try: content = raw_bytes.decode(enc); break
            except: continue
        if not content:
            flash('Cannot decode file. Save as UTF-8 CSV.', 'danger')
            return redirect(url_for('import_csv'))

        stream = io.StringIO(content)
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            flash('File is empty.', 'danger')
            return redirect(url_for('import_csv'))

        norm = {k.strip().lower().replace(' ','_').replace('-','_'): k for k in reader.fieldnames}
        REQUIRED = {'name','age','gender','purchase_freq','total_spending','last_purchase_date'}
        missing  = REQUIRED - set(norm.keys())
        if missing:
            flash(f'Missing columns: {", ".join(sorted(missing))}', 'danger')
            return redirect(url_for('import_csv'))

        def get(row, k): return row.get(norm[k],'').strip()

        added, errors = 0, 0
        for i, row in enumerate(reader, 2):
            raw_name = get(row,'name') or f'Row {i}'
            try:
                name     = get(row,'name')
                gender   = get(row,'gender').capitalize()
                age      = int(float(get(row,'age')))
                pfreq    = int(float(get(row,'purchase_freq')))
                spending = float(get(row,'total_spending').replace(',',''))
                date_str = get(row,'last_purchase_date')
                tags     = get(row,'tags') if 'tags' in norm else ''

                if not name:        raise ValueError("Name is empty")
                if gender not in ('Male','Female'): raise ValueError(f"Gender must be Male/Female")
                if not (18<=age<=100): raise ValueError(f"Age {age} out of range")

                lpdate = None
                for fmt in ('%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%d-%m-%Y'):
                    try: lpdate=datetime.strptime(date_str,fmt).date(); break
                    except: continue
                if not lpdate: raise ValueError(f"Bad date: {date_str}")

                X_raw, days_since = build_features(age, gender, pfreq, spending, lpdate)
                res    = predict_all(X_raw)
                clv    = calc_clv(pfreq, spending, days_since)
                status = 'At Risk' if res['churn_rf']=='Yes' else 'Active'

                c = Customer(
                    user_id=current_user.id, name=name, age=age, gender=gender,
                    purchase_freq=pfreq, total_spending=spending,
                    last_purchase_date=lpdate, status=status, tags=tags, clv_score=clv,
                    churn_lr=res['churn_lr'],    churn_prob_lr=res['churn_prob_lr'],
                    churn_rf=res['churn_rf'],    churn_prob_rf=res['churn_prob_rf'],
                    churn_xgb=res['churn_xgb'], churn_prob_xgb=res['churn_prob_xgb'],
                    value_lr=res['value_lr'],    value_rf=res['value_rf'],
                    value_xgb=res['value_xgb'], segment=res['segment'],
                    shap_json=json.dumps(res['shap']),
                )
                db.session.add(c)
                db.session.flush()
                db.session.add(PredictionHistory(
                    customer_id=c.id, churn_prob_rf=res['churn_prob_rf'],
                    churn_rf=res['churn_rf'], value_rf=res['value_rf'],
                    segment=res['segment'], clv_score=clv,
                ))
                added += 1
                results.append({'row':i,'name':name,'status':'OK',
                                 'churn':res['churn_rf'],'value':res['value_rf'],
                                 'segment':res['segment'],'clv':clv})
            except Exception as e:
                errors += 1
                results.append({'row':i,'name':raw_name,'status':f'Error: {e}',
                                 'churn':'—','value':'—','segment':'—','clv':'—'})

        db.session.commit()
        log_action(f'Bulk import: {added} added, {errors} errors')
        if added==0 and errors>0:
            flash(f'All {errors} rows failed. Check errors below.', 'danger')
        elif errors>0:
            flash(f'{added} imported, {errors} errors.', 'warning')
        else:
            flash(f'✓ {added} customers imported!', 'success')

    return render_template('import_csv.html', results=results)

# ══════════════════════════════════════════════════════════════════════════════
#  PDF REPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/customer/<int:cid>/pdf')
@login_required
def export_pdf(cid):
    c         = my_customers().filter_by(id=cid).first_or_404()
    shap_data = json.loads(c.shap_json or '{}')
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm,  bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    title_s = ParagraphStyle('T', parent=styles['Title'], fontSize=20, spaceAfter=6)
    story.append(Paragraph("CustPredict — Customer Report", title_s))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  By: {current_user.username}", styles['Normal']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Customer Profile", styles['Heading2']))
    t = Table([
        ['Name',c.name],['Age',str(c.age)],['Gender',c.gender],
        ['Status',c.status],['Tags',c.tags or '—'],
        ['Purchase Freq',f"{c.purchase_freq}/yr"],
        ['Total Spending',f"${c.total_spending:.2f}"],
        ['Last Purchase',str(c.last_purchase_date)],
        ['Segment',c.segment],['CLV Score',f"{c.clv_score:.1f}/100"],
    ], colWidths=[5*cm,10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#0d1520')),
        ('TEXTCOLOR',(0,0),(0,-1),colors.white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(1,0),(1,-1),[colors.whitesmoke,colors.white]),
        ('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),('PADDING',(0,0),(-1,-1),6),
    ]))
    story.append(t); story.append(Spacer(1,0.5*cm))
    story.append(Paragraph("Predictions", styles['Heading2']))
    t2 = Table([
        ['Model','Churn','Prob %','Value'],
        ['Logistic Reg.',c.churn_lr,f"{c.churn_prob_lr:.1f}%",c.value_lr],
        ['Random Forest',c.churn_rf,f"{c.churn_prob_rf:.1f}%",c.value_rf],
        ['XGBoost',c.churn_xgb,f"{c.churn_prob_xgb:.1f}%",c.value_xgb],
    ], colWidths=[5*cm,3*cm,3.5*cm,3.5*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4f8ef7')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),('PADDING',(0,0),(-1,-1),6),
    ]))
    story.append(t2)
    if shap_data:
        story.append(Spacer(1,0.5*cm))
        story.append(Paragraph("SHAP Contributions", styles['Heading2']))
        rows = [['Feature','SHAP','Direction']] + [
            [k,f"{v:+.4f}",'↑ Increases churn' if v>0 else '↓ Reduces churn']
            for k,v in sorted(shap_data.items(),key=lambda x:abs(x[1]),reverse=True)
        ]
        t3 = Table(rows, colWidths=[5*cm,4*cm,6*cm])
        t3.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1520')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
            ('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),('PADDING',(0,0),(-1,-1),5),
        ]))
        story.append(t3)
    doc.build(story)
    buf.seek(0)
    log_action(f'PDF for customer {cid}')
    return Response(buf, mimetype='application/pdf',
                    headers={'Content-Disposition':f'attachment; filename=customer_{cid}.pdf'})

# ── Dashboard PDF ─────────────────────────────────────────────────────────────
@app.route('/export/dashboard-pdf')
@login_required
def export_dashboard_pdf():
    custs = my_customers().order_by(Customer.created_at.desc()).all()
    total = len(custs)
    buf   = io.BytesIO()
    doc   = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    story.append(Paragraph("CustPredict — Dashboard Summary Report", styles['Title']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')}  |  User: {current_user.username}", styles['Normal']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1,0.4*cm))
    # KPIs
    churn_yes = sum(1 for c in custs if c.churn_rf=='Yes')
    avg_spend = round(sum(c.total_spending for c in custs)/total,2) if total else 0
    avg_clv   = round(sum(c.clv_score for c in custs)/total,1) if total else 0
    kpi_data  = [
        ['Total Customers','Churn Risk','Avg Spend','Avg CLV Score'],
        [str(total), str(churn_yes), f"${avg_spend}", f"{avg_clv}/100"],
    ]
    kt = Table(kpi_data)
    kt.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4f8ef7')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),12),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('PADDING',(0,0),(-1,-1),10),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor('#f0f4ff')),
    ]))
    story.append(kt); story.append(Spacer(1,0.5*cm))
    # Top customers
    story.append(Paragraph("Top 10 Customers", styles['Heading2']))
    rows = [['Name','Age','Spending','CLV','Churn RF','Value','Segment']]
    top  = sorted(custs, key=lambda c: c.clv_score, reverse=True)[:10]
    for c in top:
        rows.append([c.name,str(c.age),f"${c.total_spending:.0f}",
                     f"{c.clv_score:.1f}",c.churn_rf,c.value_rf,c.segment])
    ct = Table(rows, colWidths=[4*cm,1.5*cm,2.5*cm,2*cm,2*cm,2*cm,2*cm])
    ct.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#0d1520')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,-1),'Helvetica'),('FONTSIZE',(0,0),(-1,-1),9),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.whitesmoke,colors.white]),
        ('GRID',(0,0),(-1,-1),0.4,colors.lightgrey),('PADDING',(0,0),(-1,-1),5),
    ]))
    story.append(ct)
    doc.build(story)
    buf.seek(0)
    log_action('Exported dashboard PDF')
    return Response(buf, mimetype='application/pdf',
                    headers={'Content-Disposition':'attachment; filename=dashboard_report.pdf'})

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/model-performance')
@login_required
def model_performance():
    return render_template('model_performance.html',
                           metrics=META['churn_metrics'],
                           feat_imp=META['feature_importance'],
                           feat_names=list(META['feature_importance'].keys()),
                           feat_vals=list(META['feature_importance'].values()))

# ══════════════════════════════════════════════════════════════════════════════
#  DATA QUALITY CHECKER
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/data-quality', methods=['GET', 'POST'])
@login_required
def data_quality():
    """
    Standalone Data Quality Checker page.
    Accepts either:
      - A CSV file upload  →  full dataset quality report
      - A manual form fill →  single-record check
    """
    report     = None
    single_qc  = None
    mode       = None

    if request.method == 'POST':
        mode = request.form.get('mode', 'csv')

        # ── CSV file mode ──────────────────────────────────────────────────────
        if mode == 'csv':
            f = request.files.get('csv_file')
            if not f or not f.filename.lower().endswith('.csv'):
                flash('Please upload a .csv file.', 'danger')
                return redirect(url_for('data_quality'))

            raw_bytes = f.stream.read()
            content   = None
            for enc in ('utf-8-sig', 'utf-8', 'latin-1', 'cp1252'):
                try: content = raw_bytes.decode(enc); break
                except: continue

            if not content:
                flash('Cannot decode file. Save as UTF-8 CSV.', 'danger')
                return redirect(url_for('data_quality'))

            report = check_csv_quality(content)
            if 'error' in report:
                flash(report['error'], 'danger')
                return redirect(url_for('data_quality'))

            log_action(f'Data quality check: {report["total_rows"]} rows, '
                       f'score={report["avg_score"]}')

        # ── Single record mode ─────────────────────────────────────────────────
        elif mode == 'single':
            name     = request.form.get('name', '').strip()
            age      = request.form.get('age', '').strip()
            gender   = request.form.get('gender', '').strip()
            pfreq    = request.form.get('purchase_freq', '').strip()
            spending = request.form.get('total_spending', '').strip()
            lpdate   = request.form.get('last_purchase_date', '').strip()

            single_qc = check_single_record(name, age, gender, pfreq, spending, lpdate)
            single_qc['inputs'] = {
                'name': name, 'age': age, 'gender': gender,
                'purchase_freq': pfreq, 'total_spending': spending,
                'last_purchase_date': lpdate,
            }

    return render_template('data_quality.html',
                           report=report, single_qc=single_qc, mode=mode)


@app.route('/api/check-quality', methods=['POST'])
@login_required
@csrf.exempt
def api_check_quality():
    """
    Live JSON endpoint — called from the predict form via JS as the user types.
    Returns quality issues for a single record without saving anything.
    """
    data = request.get_json() or {}
    result = check_single_record(
        name              = data.get('name', ''),
        age               = data.get('age', ''),
        gender            = data.get('gender', ''),
        purchase_freq     = data.get('purchase_freq', ''),
        total_spending    = data.get('total_spending', ''),
        last_purchase_date= data.get('last_purchase_date', ''),
    )
    return jsonify(result)


# ══════════════════════════════════════════════════════════════════════════════
#  CUSTOMER COMPARISON TOOL
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/compare', methods=['GET'])
@login_required
def compare():
    """
    Compare two customers side by side.
    Query params: ?a=<id>&b=<id>
    If params missing, show a selection page.
    """
    all_customers = my_customers().order_by(Customer.name).all()

    id_a = request.args.get('a', type=int)
    id_b = request.args.get('b', type=int)

    cust_a = cust_b = None
    shap_a = shap_b = {}
    hist_a = hist_b = []
    diff   = {}

    if id_a and id_b:
        cust_a = my_customers().filter_by(id=id_a).first_or_404()
        cust_b = my_customers().filter_by(id=id_b).first_or_404()

        shap_a = json.loads(cust_a.shap_json or '{}')
        shap_b = json.loads(cust_b.shap_json or '{}')

        # Prediction histories
        ph_a = PredictionHistory.query.filter_by(customer_id=id_a)\
                                      .order_by(PredictionHistory.created_at).all()
        ph_b = PredictionHistory.query.filter_by(customer_id=id_b)\
                                      .order_by(PredictionHistory.created_at).all()

        hist_a = [{'date': p.created_at.strftime('%d %b'),
                   'prob': p.churn_prob_rf, 'clv': p.clv_score} for p in ph_a]
        hist_b = [{'date': p.created_at.strftime('%d %b'),
                   'prob': p.churn_prob_rf, 'clv': p.clv_score} for p in ph_b]

        # Compute differences (positive = A is higher)
        def pct_diff(a_val, b_val):
            if b_val == 0:
                return 0
            return round(((a_val - b_val) / abs(b_val)) * 100, 1)

        diff = {
            'age':           cust_a.age          - cust_b.age,
            'purchase_freq': cust_a.purchase_freq - cust_b.purchase_freq,
            'total_spending':round(cust_a.total_spending - cust_b.total_spending, 2),
            'churn_prob_rf': round(cust_a.churn_prob_rf  - cust_b.churn_prob_rf, 1),
            'clv_score':     round(cust_a.clv_score      - cust_b.clv_score, 1),
            'churn_prob_rf_pct': pct_diff(cust_a.churn_prob_rf, cust_b.churn_prob_rf),
            'clv_pct':           pct_diff(cust_a.clv_score,     cust_b.clv_score),
            'spending_pct':      pct_diff(cust_a.total_spending, cust_b.total_spending),
        }

        # Shared SHAP feature names
        all_features = list(set(list(shap_a.keys()) + list(shap_b.keys())))

        log_action(f'Compared customers {id_a} vs {id_b}')

        return render_template('compare.html',
                               all_customers=all_customers,
                               cust_a=cust_a, cust_b=cust_b,
                               shap_a=shap_a, shap_b=shap_b,
                               hist_a=hist_a, hist_b=hist_b,
                               diff=diff, all_features=all_features,
                               id_a=id_a, id_b=id_b)

    # No IDs yet — show picker
    return render_template('compare.html',
                           all_customers=all_customers,
                           cust_a=None, cust_b=None,
                           id_a=id_a, id_b=id_b)


# ══════════════════════════════════════════════════════════════════════════════
#  ACCOUNT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/account', methods=['GET','POST'])
@login_required
def account():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'update_profile':
            current_user.full_name = request.form.get('full_name','').strip()
            new_email = request.form.get('email','').strip().lower()
            if new_email != current_user.email and User.query.filter_by(email=new_email).first():
                flash('Email already in use.','danger')
            else:
                current_user.email = new_email
                db.session.commit()
                log_action('Updated profile')
                flash('Profile updated.','success')

        elif action == 'upload_avatar':
            file = request.files.get('avatar')
            if file and allowed_file(file.filename):
                ext  = file.filename.rsplit('.',1)[1].lower()
                fname = f"avatar_{current_user.id}.{ext}"
                path  = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                file.save(path)
                current_user.avatar = fname
                db.session.commit()
                flash('Profile photo updated.','success')
            else:
                flash('Invalid file. Use PNG, JPG, GIF or WEBP.','danger')

        elif action == 'change_password':
            old = request.form.get('old_password','')
            new = request.form.get('new_password','')
            if not current_user.check_password(old):
                flash('Current password incorrect.','danger')
            elif len(new) < 6:
                flash('New password must be at least 6 characters.','danger')
            else:
                current_user.set_password(new)
                db.session.commit()
                log_action('Changed password')
                flash('Password updated.','success')

        elif action == 'regen_api_key':
            current_user.generate_api_key()
            db.session.commit()
            log_action('Regenerated API key')
            flash('New API key generated.','success')

        elif action == 'delete_account':
            if request.form.get('confirm_delete') == current_user.username:
                username = current_user.username
                logout_user()
                u = User.query.filter_by(username=username).first()
                db.session.delete(u)
                db.session.commit()
                flash('Account permanently deleted.','info')
                return redirect(url_for('register'))
            flash('Username confirmation did not match.','danger')

    total_customers = my_customers().count()
    return render_template('account.html', total_customers=total_customers)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ══════════════════════════════════════════════════════════════════════════════
#  AI CHATBOT
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/chatbot')
@login_required
def chatbot():
    # Get user's summary stats to give the AI context
    custs     = my_customers().all()
    total     = len(custs)
    churn_yes = sum(1 for c in custs if c.churn_rf=='Yes')
    avg_clv   = round(sum(c.clv_score for c in custs)/total,1) if total else 0
    high_val  = sum(1 for c in custs if c.value_rf=='High')
    context   = {
        'total': total, 'churn_yes': churn_yes,
        'avg_clv': avg_clv, 'high_value': high_val,
    }
    return render_template('chatbot.html', context=context)

@app.route('/api/chat', methods=['POST'])
@login_required
@csrf.exempt
def api_chat():
    """
    Calls Anthropic Claude API with user's data context.
    The chatbot knows the user's customer stats and can answer
    business questions about churn, CLV, segments etc.
    """
    data        = request.get_json()
    user_msg    = data.get('message','').strip()
    history     = data.get('history', [])

    if not user_msg:
        return jsonify({'error': 'Empty message'}), 400

    # Build data context for the AI
    custs     = my_customers().all()
    total     = len(custs)
    churn_yes = sum(1 for c in custs if c.churn_rf=='Yes')
    avg_clv   = round(sum(c.clv_score for c in custs)/total,1) if total else 0
    high_val  = sum(1 for c in custs if c.value_rf=='High')
    avg_spend = round(sum(c.total_spending for c in custs)/total,2) if total else 0

    # Top 5 at-risk customers
    at_risk = sorted([c for c in custs if c.churn_rf=='Yes'],
                     key=lambda c: c.churn_prob_rf, reverse=True)[:5]
    at_risk_str = ', '.join([f"{c.name} ({c.churn_prob_rf:.0f}%)" for c in at_risk]) or 'None'

    system_prompt = f"""You are CustBot, an expert AI business analyst embedded inside CustPredict — 
an AI-powered customer performance prediction system.

You have access to the following LIVE data for the current user ({current_user.username}):

📊 CUSTOMER SUMMARY:
- Total customers: {total}
- Churn risk (RF model): {churn_yes} customers ({round(churn_yes/total*100,1) if total else 0}%)
- High-value customers: {high_val}
- Average CLV score: {avg_clv}/100
- Average spending: ${avg_spend}

⚠️ TOP AT-RISK CUSTOMERS: {at_risk_str}

🤖 ML MODELS USED:
- Logistic Regression, Random Forest, XGBoost for churn & value prediction
- K-Means clustering for segmentation (Gold/Silver/Bronze)
- SHAP for explainability
- CLV score = weighted recency (30%) + frequency (35%) + monetary (35%)

Your job:
- Answer questions about the user's customers, churn risk, CLV, segments
- Give actionable business advice (e.g. "which customers to focus on?")
- Explain ML concepts in simple terms when asked
- Suggest strategies to reduce churn or increase customer value
- Be concise, data-driven, and helpful
- Use the data above to give personalised answers
- If asked something outside your scope, politely redirect

Always respond in plain text (no markdown). Be friendly and professional."""

    # Build messages for API
    messages = []
    for h in history[-10:]:  # keep last 10 turns
        messages.append({'role': h['role'], 'content': h['content']})
    messages.append({'role': 'user', 'content': user_msg})

    try:
        import urllib.request
        groq_key = os.environ.get('GROQ_API_KEY', '')
        if not groq_key:
            return jsonify({'reply': "Chatbot is not configured. Please set the GROQ_API_KEY environment variable."}), 200

        payload = json.dumps({
            'model':      'llama-3.3-70b-versatile',
            'max_tokens': 600,
            'messages':   [{'role': 'system', 'content': system_prompt}] + messages,
        }).encode('utf-8')

        import http.client
        import ssl

        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.groq.com", context=ctx)
        conn.request(
            "POST",
            "/openai/v1/chat/completions",
            body=payload,
            headers={
                'Content-Type':  'application/json',
                'Authorization': f'Bearer {groq_key}',
                'User-Agent':    'python-httpx/0.27.0',
            }
        )
        resp   = conn.getresponse()
        result = json.loads(resp.read().decode('utf-8'))
        conn.close()
        if 'choices' not in result:
            err = result.get('error', {}).get('message', str(result))
            return jsonify({'reply': f"API error: {err}"}), 200
        reply  = result['choices'][0]['message']['content']
        return jsonify({'reply': reply})

    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='ignore')
        try:
            err = json.loads(body).get('error', {}).get('message', str(e)) if body else str(e)
        except Exception:
            err = body if body else str(e)
        return jsonify({'reply': f"I'm having trouble connecting right now. Error: {err}"}), 200
    except Exception as e:
        return jsonify({'reply': f"Sorry, I encountered an error: {str(e)}"}), 200

# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    all_users = User.query.order_by(User.created_at).all()
    return render_template('admin_users.html', user_stats=[
        {'user':u, 'customer_count': Customer.query.filter_by(user_id=u.id).count()}
        for u in all_users
    ])

@app.route('/admin/users/<int:uid>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_user(uid):
    u = db.get_or_404(User, uid)
    if u.id == current_user.id:
        flash("Can't delete your own account here.", 'danger')
    else:
        name = u.username
        db.session.delete(u)
        db.session.commit()
        log_action(f'Admin deleted user: {name}')
        flash(f'User "{name}" deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/audit-log')
@login_required
@admin_required
def audit_log():
    logs = AuditLog.query.order_by(AuditLog.timestamp.desc()).limit(300).all()
    return render_template('audit_log.html', logs=logs)

# ══════════════════════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/customers')
@login_required
def api_customers():
    return jsonify([{
        'id':c.id,'name':c.name,'age':c.age,'gender':c.gender,
        'churn_rf':c.churn_rf,'churn_prob_rf':c.churn_prob_rf,
        'value_rf':c.value_rf,'segment':c.segment,
        'clv_score':c.clv_score,'status':c.status,
    } for c in my_customers().order_by(Customer.created_at.desc()).all()])

@app.route('/api/predict', methods=['POST'])
@login_required
@csrf.exempt
def api_predict():
    data = request.get_json()
    try:
        lpdate        = datetime.strptime(data['last_purchase_date'],'%Y-%m-%d').date()
        X_raw, days   = build_features(int(data['age']),data['gender'],
                                        int(data['purchase_freq']),
                                        float(data['total_spending']),lpdate)
        result        = predict_all(X_raw)
        result['clv'] = calc_clv(int(data['purchase_freq']),float(data['total_spending']),days)
        return jsonify({'status':'ok','predictions':result})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}),400

@app.route('/api/stats')
@login_required
def api_stats():
    q     = my_customers()
    total = q.count()
    avg   = db.session.query(db.func.avg(Customer.total_spending))\
              .filter_by(user_id=current_user.id).scalar() or 0
    return jsonify({
        'total':total,
        'churn': q.filter_by(churn_rf='Yes').count(),
        'high_value': q.filter_by(value_rf='High').count(),
        'avg_spend': round(float(avg),2),
    })

# ── DB init ────────────────────────────────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        print("[✓] Database ready.")
        print("    Visit http://127.0.0.1:5000/register")
        print("    First user to register = admin.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
