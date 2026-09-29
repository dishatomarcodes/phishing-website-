import sys, os, argparse, threading, webbrowser

# Auto-install any missing packages before importing anything else
def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

REQUIRED = {
    "flask":      "flask",
    "flask_cors": "flask-cors",
    "sklearn":    "scikit-learn",
    "numpy":      "numpy",
    "tldextract": "tldextract",
    "joblib":     "joblib",
}
for import_name, pip_name in REQUIRED.items():
    try:
        __import__(import_name)
    except ImportError:
        print(f"Installing {pip_name}...")
        install(pip_name)


from classifier import classify
from server import app


def print_result(url):
    """CLI output — prints a clean summary to the terminal."""
    r    = classify(url)
    icon = {"Safe": "✅", "Suspicious": "⚠️", "Phishing": "🚨"}[r["level"]]
    p    = r["probabilities"]

    print()
    print("=" * 58)
    print(f"  URL     : {r['url']}")
    print(f"  Result  : {icon}  {r['level']}   (phishing score: {r['score']})")
    print(f"  Chances : Safe {p['Safe']}  |  Suspicious {p['Suspicious']}  |  Phishing {p['Phishing']}")
    print()
    print("  Reasons:")
    for reason in r["explanation"]:
        print(f"    • {reason}")
    print("=" * 58)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing URL Classifier")
    parser.add_argument("--url",  help="Classify a single URL and print result (no server)")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.url:
        # CLI mode
        print_result(args.url)
    else:
        # GUI mode — start server and open browser
        port = args.port
        print(f"\n  Starting Phishing Classifier...")
        print(f"  Open your browser at:  http://localhost:{port}")
        print(f"  Press Ctrl+C to stop.\n")
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()
        app.run(host="0.0.0.0", port=port, debug=False)
