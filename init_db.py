"""
init_db.py  –  v4
-----------------
Creates DB tables and seeds a default admin account.
Default credentials:
    Username : admin
    Password : admin123
Everyone else can self-register at /register.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from app import app, db, User, Customer, build_features, predict_all, calc_clv, PredictionHistory, TimelineEvent
from datetime import date, timedelta

SAMPLE_CUSTOMERS = [
    ("Priya Sharma",    28, "Female", 24, 2800,  15),
    ("Rahul Gupta",     35, "Male",   10,  450, 200),
    ("Anita Patel",     42, "Female", 38, 4500,   5),
    ("Vikram Singh",    55, "Male",    4,  120, 310),
    ("Kavita Rao",      31, "Female", 19, 1750,  30),
    ("Suresh Kumar",    48, "Male",   29, 3200,  12),
    ("Meena Nair",      26, "Female",  7,  300, 180),
    ("Arun Joshi",      39, "Male",   45, 4900,   3),
    ("Deepa Menon",     33, "Female", 15, 1100,  60),
    ("Rajesh Verma",    52, "Male",    2,   80, 350),
]

def init():
    with app.app_context():
        db.create_all()

        # ── Create default admin ───────────────────────────────────────────────
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username  = 'admin',
                email     = 'admin@custpredict.com',
                full_name = 'System Admin',
                role      = 'admin',
            )
            admin.set_password('admin123')
            admin.generate_api_key()
            db.session.add(admin)
            db.session.flush()   # get admin.id before commit
            print("[✓] Default admin created")
            print("    Username : admin")
            print("    Password : admin123")

            # ── Seed sample customers for the admin account ────────────────────
            added = 0
            for name, age, gender, pfreq, spending, days_ago in SAMPLE_CUSTOMERS:
                lpdate = date.today() - timedelta(days=days_ago)
                try:
                    X_raw, days_since = build_features(age, gender, pfreq, float(spending), lpdate)
                    res    = predict_all(X_raw)
                    clv    = calc_clv(pfreq, float(spending), days_since)
                    status = 'At Risk' if res['churn_rf'] == 'Yes' else 'Active'

                    c = Customer(
                        user_id            = admin.id,
                        name               = name,
                        age                = age,
                        gender             = gender,
                        purchase_freq      = pfreq,
                        total_spending     = float(spending),
                        last_purchase_date = lpdate,
                        status             = status,
                        clv_score          = clv,
                        churn_lr           = res['churn_lr'],
                        churn_prob_lr      = res['churn_prob_lr'],
                        churn_rf           = res['churn_rf'],
                        churn_prob_rf      = res['churn_prob_rf'],
                        churn_xgb          = res['churn_xgb'],
                        churn_prob_xgb     = res['churn_prob_xgb'],
                        value_lr           = res['value_lr'],
                        value_rf           = res['value_rf'],
                        value_xgb          = res['value_xgb'],
                        segment            = res['segment'],
                        shap_json          = json.dumps(res['shap']),
                    )
                    db.session.add(c)
                    db.session.flush()

                    db.session.add(PredictionHistory(
                        customer_id   = c.id,
                        churn_prob_rf = res['churn_prob_rf'],
                        churn_rf      = res['churn_rf'],
                        value_rf      = res['value_rf'],
                        segment       = res['segment'],
                        clv_score     = clv,
                    ))
                    db.session.add(TimelineEvent(
                        customer_id = c.id,
                        event_type  = 'created',
                        description = f'Customer added with churn risk {res["churn_prob_rf"]}%',
                    ))
                    added += 1
                except Exception as e:
                    print(f"  ✗ Skipped {name}: {e}")

            db.session.commit()
            print(f"[✓] Seeded {added} sample customers for admin account")

        else:
            print("[i] Admin account already exists — skipping seed")

        print()
        print("─" * 45)
        print("  DEFAULT LOGIN CREDENTIALS")
        print("─" * 45)
        print("  Username : admin")
        print("  Password : admin123")
        print("─" * 45)
        print()
        print("  Run:  python app.py")
        print("  Open: http://127.0.0.1:5000")
        print()
        print("  Others can register at /register")
        print("  Every user sees only their own data")

if __name__ == '__main__':
    init()
