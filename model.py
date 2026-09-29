# ============================================================
#  Layer 3 — Model
#  Labels: 0 = Safe, 1 = Suspicious, 2 = Phishing
# ============================================================

import os, joblib
from sklearn.ensemble import GradientBoostingClassifier
from features import get_features

MODEL_FILE = os.path.join(os.path.dirname(__file__), "phish_model.joblib")
LABELS = {0: "Safe", 1: "Suspicious", 2: "Phishing"}

# ── Training data ─────────────────────────────────────────
# 0 = Safe   1 = Suspicious   2 = Phishing
TRAINING_EXAMPLES = [

    # ── SAFE (0) ─────────────────────────────────────────
    ("https://www.google.com",                                     0),
    ("https://www.google.com/search?q=python",                     0),
    ("https://mail.google.com/mail/u/0/",                          0),
    ("https://github.com/openai/openai-python",                    0),
    ("https://github.com/torvalds/linux",                          0),
    ("https://stackoverflow.com/questions/tagged/python",          0),
    ("https://www.wikipedia.org/wiki/Phishing",                    0),
    ("https://docs.python.org/3/library/urllib.parse.html",        0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ",               0),
    ("https://www.bbc.com/news/technology",                        0),
    ("https://www.apple.com/iphone",                               0),
    ("https://www.microsoft.com/en-us/windows",                    0),
    ("https://www.linkedin.com/in/example",                        0),
    ("https://www.reddit.com/r/python",                            0),
    ("https://www.cloudflare.com/learning/ddos/what-is-a-ddos-attack", 0),
    ("https://www.nasa.gov/missions",                              0),
    ("https://www.nhs.uk/conditions",                              0),
    ("https://aws.amazon.com/s3/",                                 0),
    ("https://azure.microsoft.com/en-us/",                         0),
    ("https://www.coursera.org/learn/machine-learning",            0),
    ("https://www.nytimes.com/section/technology",                 0),
    ("https://www.bbc.co.uk/sport",                                0),
    ("https://www.instagram.com/nasa/",                            0),
    ("https://twitter.com/elonmusk",                               0),
    ("https://www.dropbox.com/home",                               0),
    ("https://zoom.us/join",                                       0),
    ("https://www.spotify.com/account/overview/",                  0),
    ("https://www.netflix.com/browse",                             0),
    ("https://www.ebay.com/itm/123456789",                         0),
    ("https://www.paypal.com/myaccount/summary",                   0),

    # ── SUSPICIOUS (1) ───────────────────────────────────
    # Real brand name used but HTTP, or sketchy structure
    ("http://login-verify.example.com/account?update=true",        1),
    ("http://secure-paypal.net/confirm",                           1),
    ("http://amazon-support.xyz/update-billing",                   1),
    ("http://free-iphone-winner.online/claim",                     1),
    ("http://microsoft-account-alert.info/signin",                 1),
    ("http://192.168.1.1/admin/login",                             1),
    ("http://apple-id-locked.site/verify",                         1),
    ("http://banking-secure.biz/update",                           1),
    ("http://urgent-account-verify.top/login",                     1),
    ("http://ebay-support-help.net/signin",                        1),
    ("http://paypal.suspicious-domain.com/login",                  1),
    ("http://google.phishing-test.net/signin",                     1),
    ("http://amazon-prime-deals.club/checkout",                    1),
    ("http://win-free-gift.online/claim?user=test",                1),
    ("http://account-locked.info/verify-now",                      1),
    ("http://secure-login-portal.xyz/auth",                        1),
    ("http://update-your-details.biz/account",                     1),
    ("http://customer-support-helpdesk.top/ticket",                1),
    ("http://limited-offer-today.click/deal",                      1),
    ("http://your-account-suspended.site/reactivate",              1),
    # Punycode / IDN (visually spoofed domains)
    ("https://xn--e1afmkfd.example.com/login",                     1),
    ("http://xn--pypal-4ve.com/login",                             1),  # pаypal
    ("http://xn--googl-fsa.com/signin",                            1),  # gооgle
    ("http://xn--amzon-tsa.com/login",                             1),  # amаzon
    ("http://xn--micosoft-6hb.com/account",                        1),  # miсrosoft

    # ── PHISHING (2) ─────────────────────────────────────
    ("http://paypa1.com-secure-login.tk/account/signin",           2),
    ("http://192.168.10.55/phish/login.php?user=victim",           2),
    ("http://goog1e-login.ml/auth?redirect=http%3A%2F%2Fevil.com",2),
    ("http://secure-banking-update.ga/login?token=abc&session=xyz",2),
    ("http://amaz0n-winner.cf/lucky?prize=gift&click=here",        2),
    ("http://verify-your-paypal.gq/confirm-identity",              2),
    ("http://facebook-support-account.tk/login?urgent=1",          2),
    ("http://apple-id-suspended.ml/reactivate?user=me@gmail.com",  2),
    ("http://update-your-banking-credentials.cf/secure/login",     2),
    ("http://microsoft-security-alert.ga/signin?redirect=evil",    2),
    ("http://ebay-account-suspended.tk/verify?session=12345",      2),
    ("http://netflix-billing-update.tk/account/payment",           2),
    ("http://login-google-accounts.cf/auth?code=abc%20xyz",        2),
    ("http://paypal-confirm-identity.gq/update-password",          2),
    ("http://secure-apple-id.ml/verify-account?redirect=phish",    2),
    ("http://amazon-gift-claim.ga/prize?user=foo@bar.com",         2),
    ("http://wellsfargo-secure.tk/signin?session=abc123",          2),
    ("http://bankofamerica-update.ml/verify-identity",             2),
    ("http://chase-alert.cf/secure-login?token=xyz",               2),
    ("http://linkedin-security.gq/account-verify",                 2),
    ("http://dropbox-fileshare.tk/download?file=invoice.exe",      2),
    ("http://instagram-verify.ml/confirm-account?code=12345",      2),
    ("http://twitter-suspended.cf/appeal?user=victim",             2),
    ("http://support-microsoft-alert.ga/remove-virus",             2),
    ("http://appleid-locked.tk/unlock?session=%3Cscript%3E",       2),
    ("http://paypal-resolution-centre.ml/dispute?id=99999",        2),
    ("http://account-google-verify.cf/two-step?backup=evil",       2),
    ("http://amazon-order-problem.gq/resolve?order=fake",          2),
    ("http://netflix-payment-failed.tk/update-card",               2),
    ("http://ebay-unpaid-item.ml/pay-now?item=99999",              2),
]


def load_or_train():
    # Always retrain if feature count changed (delete old model to force retrain)
    if os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            # Validate feature count matches current extractor
            import numpy as np
            from features import get_features
            test_vec = list(get_features("https://example.com").values())
            m.predict([test_vec])   # will raise if shape mismatch
            return m
        except Exception:
            print("Feature shape changed — retraining model...")
            os.remove(MODEL_FILE)

    print("Training model...")
    from features import get_features
    X = [list(get_features(url).values()) for url, _ in TRAINING_EXAMPLES]
    y = [label for _, label in TRAINING_EXAMPLES]

    # GradientBoosting generalises better than Random Forest on small datasets
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
        subsample=0.9,
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Model trained and saved.\n")
    return model


MODEL = load_or_train()
