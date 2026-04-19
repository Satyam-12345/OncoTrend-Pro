from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max
import json
import os

# Set Hadoop path (IMPORTANT)
os.environ['HADOOP_HOME'] = 'C:\\hadoop'
os.environ['PATH'] += ';C:\\hadoop\\bin'

# Dataset Configuration
CONFIG = {
    "Breast Cancer": {
        "file": "hdfs://localhost:9000/user/hadoop/raw/breast_cancer.csv",
        "features": [
            ("radius1", "Radius Mean", "mm"),
            ("texture1", "Texture Mean", "gray"),
            ("perimeter1", "Perim. Mean", "mm"),
            ("area1", "Area Mean", "mm²"),
            ("smoothness1", "Smooth. Mean", "ratio")
        ],
        "target": "Diagnosis",
        "label_map": {"M": "Malignant", "B": "Benign"}
    },
    "Cervical Cancer": {
        "file": "hdfs://localhost:9000/user/hadoop/raw/cervical_cancer.csv",
        "features": [
            ("Age", "Patient Age", "yrs"),
            ("Number of sexual partners", "Partners", "count"),
            ("Num of pregnancies", "Pregnancies", "count"),
            ("Smokes (years)", "Smoking Yrs", "yrs")
        ],
        "target": "Biopsy",
        "label_map": {1: "High Risk", 0: "Low Risk"}
    },
    "Lung Cancer": {
        "file": "hdfs://localhost:9000/user/hadoop/raw/lung_cancer.csv",
        "features": [
            ("Attribute1", "Lung Var A", "idx"),
            ("Attribute2", "Lung Var B", "idx"),
            ("Attribute3", "Lung Var C", "idx"),
            ("Attribute4", "Lung Var D", "idx"),
            ("Attribute5", "Lung Var E", "idx")
        ],
        "target": "class",
        "label_map": {1: "Type 1", 2: "Type 2", 3: "Type 3"}
    },
    "Chronic Kidney Disease": {
        "file": "hdfs://localhost:9000/user/hadoop/raw/chronic_kidney_disease.csv",
        "features": [
            ("age", "Patient Age", "yrs"),
            ("bp", "Blood Pressure", "mmHg"),
            ("bgr", "Blood Glucose", "mgs/dl"),
            ("bu", "Blood Urea", "mgs/dl"),
            ("sc", "Serum Creatinine", "mgs/dl")
        ],
        "target": "class",
        "label_map": {"ckd": "Positive", "notckd": "Negative"}
    }
}


def run_advanced_analytics():

    # Create Spark session ONCE
    spark = spark = SparkSession.builder \
                    .appName("Clinical-Hadoop") \
                    .master("local") \
                    .config("spark.driver.host", "127.0.0.1") \
                    .config("spark.driver.bindAddress", "127.0.0.1") \
                    .getOrCreate()

    all_registry = {}

    for name, cfg in CONFIG.items():
        print(f"Processing: {name}...")

        file_path = cfg["file"]
        df = spark.read.csv(file_path, header=True, inferSchema=True)

        feature_cols = [f[0] for f in cfg["features"]]

        # Aggregations
        agg_exprs = []
        for c in feature_cols:
            agg_exprs.append(mean(c).alias(f"{c}_avg"))
            agg_exprs.append(stddev(c).alias(f"{c}_std"))
            agg_exprs.append(spark_min(c).alias(f"{c}_min"))
            agg_exprs.append(spark_max(c).alias(f"{c}_max"))

        stats = df.groupBy(cfg["target"]).agg(*agg_exprs).collect()

        results = {}

        for row in stats:
            raw_label = str(row[cfg["target"]])
            label = cfg["label_map"].get(
                raw_label,
                cfg["label_map"].get(int(raw_label) if raw_label.isdigit() else raw_label, raw_label)
            )

            feature_stats = {}

            for col_name, display, unit in cfg["features"]:
                feature_stats[display] = {
                    "mean": float(row[f"{col_name}_avg"]),
                    "std": float(row[f"{col_name}_std"]) if row[f"{col_name}_std"] else 0.1,
                    "min": float(row[f"{col_name}_min"]),
                    "max": float(row[f"{col_name}_max"]),
                    "unit": unit
                }

            results[label] = feature_stats

        all_registry[name] = {
            "baselines": results,
            "features": cfg["features"]
        }

        print(f"{name} processed successfully.")

    spark.stop()

    # Save locally
    local_file = "clinical_baselines.json"
    with open(local_file, "w") as f:
        json.dump(all_registry, f, indent=4)

    # Upload to HDFS
    os.system(f"hdfs dfs -put -f {local_file} /user/hadoop/processed/")

    print("Baselines uploaded to HDFS successfully")


if __name__ == "__main__":
    run_advanced_analytics()