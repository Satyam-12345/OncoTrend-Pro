from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# HDFS Path for Processed Registry
from hdfs import InsecureClient

client = InsecureClient('http://localhost:9870', user='hadoop')

BASELINE_FILE = "clinical_baselines.json"

def get_baselines():
    try:
        with open(BASELINE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error:", e)
        return None

@app.route("/")
def index():
    registry = get_baselines()
    domains = list(registry.keys()) if registry else []
    return render_template("index.html", domains=domains)

@app.route("/get_meta")
def get_meta():
    domain = request.args.get("domain")
    registry = get_baselines()
    if not registry or domain not in registry:
        return jsonify({"error": "Invalid domain"}), 400
    
    # Extract units and ranges from the first available baseline for this domain
    first_label = list(registry[domain]["baselines"].keys())[0]
    feature_meta = {}
    for col_name, display, unit in registry[domain]["features"]:
        feature_meta[display] = {
            "unit": unit,
            "min": registry[domain]["baselines"][first_label][display]["min"],
            "max": registry[domain]["baselines"][first_label][display]["max"]
        }
    
    return jsonify({
        "features": registry[domain]["features"],
        "metadata": feature_meta,
        "labels": list(registry[domain]["baselines"].keys())
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    domain = data.get("domain")
    user_inputs = data.get("inputs", {})

    registry = get_baselines()
    if not registry or domain not in registry:
        return jsonify({"error": "System registry synchronized... checking baselines"}), 500

    domain_data = registry[domain]
    baselines = domain_data["baselines"]
    features_meta = domain_data["features"]

    interpretations = []
    radar_data = {
        "patient": [],
        "baselines": {label: [] for label in baselines.keys()},
        "labels": [f[1] for f in features_meta]
    }

    # Clinical Significance Analysis (Z-Score)
    risk_scores = {label: 0 for label in baselines.keys()}

    for col_name, display_name, unit in features_meta:
        try:
            u_val = float(user_inputs.get(display_name, 0))
        except ValueError:
            u_val = 0.0
            
        radar_data["patient"].append(u_val)
        
        closest_label = None
        min_z = float('inf')
        
        for label, stats in baselines.items():
            mean = stats[display_name]["mean"]
            std = stats[display_name]["std"] or 0.1
            
            # Z-Score: distance from norm in standard deviations
            z = abs(u_val - mean) / std
            
            radar_data["baselines"][label].append(mean)
            
            if z < min_z:
                min_z = z
                closest_label = label
        
        if closest_label:
            risk_scores[closest_label] += 1
        
        interpretations.append({
            "feature": display_name,
            "closest": closest_label or "Undetermined",
            "z_score": round(min_z, 2) if min_z != float('inf') else 0.0
        })

    # Final Diagnostic Logic
    diagnosis = max(risk_scores, key=risk_scores.get)
    
    # Aesthetic mapping
    theme = {
        "Positive": "#ff4d4d", "Malignant": "#ff4d4d", "High Risk": "#ff4d4d",
        "Negative": "#4dff88", "Benign": "#4dff88", "Low Risk": "#4dff88"
    }
    color = theme.get(diagnosis, "#38bdf8")

    return jsonify({
        "diagnosis": diagnosis,
        "color": color,
        "interpretations": interpretations,
        "radar": radar_data
    })

if __name__ == "__main__":
    app.run(debug=True, port=5001)
