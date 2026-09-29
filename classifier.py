# ============================================================
#  Layer 4 — Classifier & Response
#
#  Two-stage decision:
#  1. Hard rules  — always-correct signals that override the model
#  2. ML model    — catches subtler patterns the rules miss
# ============================================================

import numpy as np
from features import get_features, BRANDS, DODGY_TLDS
from model import MODEL, LABELS


# ── Hard rule engine ──────────────────────────────────────────────────────────

def hard_classify(f, url):
    """
    Returns (label_int, [reasons]) if a hard rule fires, else None.
    These are high-confidence signals that don't need the model.
    """
    reasons = []
    score = 0  # accumulate suspicion points

    # ── Definite Phishing ──
    if f["uses_ip_address"]:
        reasons.append("Uses a raw IP address as the domain — a strong phishing indicator.")
        score += 3
    if f["has_punycode"]:
        reasons.append("Domain uses punycode (xn--) encoding — commonly used in homograph attacks to visually mimic real brands.")
        score += 3
    if f["has_at_symbol"]:
        reasons.append("Contains '@' in the URL — tricks browsers into ignoring the real destination domain.")
        score += 2
    if f["dodgy_tld"]:
        reasons.append(f"Domain extension is commonly abused for phishing (.tk, .ml, .ga, .cf, .gq, etc.).")
        score += 2
    if f["brand_in_subdomain"]:
        reasons.append("A well-known brand name appears in the subdomain — a classic spoofing technique.")
        score += 2
    if f["typosquat_hyphen"]:
        reasons.append("Brand name used with hyphens in the domain — typical of typosquatting attacks.")
        score += 2
    if f["has_hex_encoding"]:
        reasons.append("Contains percent-encoded (hex) characters — often used to obfuscate the real URL.")
        score += 1
    if f["phishing_word_count"] >= 3:
        reasons.append(f"Contains {f['phishing_word_count']} phishing-related keywords (login, verify, secure, etc.).")
        score += 2
    elif f["phishing_word_count"] == 2:
        reasons.append("Contains multiple phishing keywords.")
        score += 1
    if f["uses_https"] == 0 and score > 0:
        reasons.append("Does not use HTTPS — connection is unencrypted.")
        score += 1
    if f["url_length"] > 120:
        reasons.append("Unusually long URL — often used to hide the real destination.")
        score += 1
    if f["dot_count"] > 4:
        reasons.append("Excessive subdomains in the hostname.")
        score += 1
    if f["hyphen_count"] > 3:
        reasons.append("Many hyphens in the domain — common in spoofed or generated domains.")
        score += 1
    if f["url_entropy"] > 5.0:
        reasons.append("High URL entropy — the address looks randomly or algorithmically generated.")
        score += 1
    if f["digit_ratio"] > 0.25:
        reasons.append("High proportion of digits in the URL.")
        score += 1

    if score >= 5:
        return 2, reasons   # Phishing
    if score >= 2:
        return 1, reasons   # Suspicious
    return None             # Let the model decide


# ── Explanation for model-decided results ─────────────────────────────────────

def explain_model(f, prediction):
    """Fallback explanations when the model (not hard rules) made the call."""
    reasons = []
    if f["uses_https"] == 0 and prediction > 0:
        reasons.append("Does not use HTTPS.")
    if f["phishing_word_count"] >= 1:
        reasons.append(f"Contains {f['phishing_word_count']} phishing-related keyword(s).")
    if f["brand_in_path"]:
        reasons.append("A known brand name appears in the URL path.")
    if f["subdomain_depth"] > 2:
        reasons.append("Unusual subdomain depth.")
    if f["url_entropy"] > 4.5:
        reasons.append("URL entropy is higher than typical legitimate sites.")
    if not reasons:
        reasons.append("No major red flags detected — URL structure appears normal.")
    return reasons


# ── Main classify function ────────────────────────────────────────────────────

def classify(url):
    f = get_features(url)
    vec = np.array([list(f.values())])

    # Stage 1: hard rules
    hard = hard_classify(f, url)

    if hard:
        pred, reasons = hard
        # Still get model probabilities for the confidence bars
        try:
            proba = MODEL.predict_proba(vec)[0]
        except Exception:
            proba = [0.0, 0.0, 0.0]
            proba[pred] = 1.0
    else:
        # Stage 2: ML model
        pred  = int(MODEL.predict(vec)[0])
        proba = MODEL.predict_proba(vec)[0]
        reasons = explain_model(f, pred)

    # Adjust probabilities to reflect hard-rule override
    if hard:
        # Blend: keep model's distribution but force the hard-rule class dominant
        p = list(proba)
        p[pred] = max(p[pred], 0.70)
        total = sum(p)
        proba = [x / total for x in p]

    return {
        "url":   url,
        "level": LABELS[pred],
        "score": round(float(proba[2]), 3),
        "probabilities": {
            LABELS[i]: round(float(p), 3)
            for i, p in enumerate(proba)
        },
        "explanation": reasons,
        "features":    f,
        "decided_by":  "rules" if hard else "model",
    }
