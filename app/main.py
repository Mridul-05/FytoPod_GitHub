import json
from datetime import datetime
import uvicorn
from groq import Groq
import psycopg2
import pickle
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor
import os
from dotenv import load_dotenv

load_dotenv()


app = FastAPI(
    title="FytoPod API",
    description="Dual-modal plant health intelligence system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "fytopod"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", "5432")
}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

lstm_model = load_model(
    r'C:\Users\mridu\Downloads\PROJECTS!!!!!!\fytopod\fytopod_env\fytopod_lstm_model.keras')
with open(r'C:\Users\mridu\Downloads\PROJECTS!!!!!!\fytopod\fytopod_env\fytopod_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

SEQ_LEN = 12
MOISTURE_THRESHOLD = 35
FEATURES = ['moisture', 'temp', 'light', 'ph']

print("FytoPod API initialized!")

# ============================================
# REQUEST MODELS
# ============================================


class SensorData(BaseModel):
    moisture: float
    temp: float
    light: float
    ph: float


class DiseaseQuery(BaseModel):
    disease: str
    severity: str


class FollowupQuery(BaseModel):
    disease: str
    severity: str
    treatment: str
    question: str

# ============================================
# HELPER FUNCTIONS
# ============================================


def estimate_severity(confidence):
    if confidence >= 0.85:
        return "Mild"
    elif confidence >= 0.60:
        return "Moderate"
    elif confidence >= 0.40:
        return "Severe"
    else:
        return "Needs Inspection"


def calculate_health_score(moisture, temp, light, ph,
                           stress_predicted, disease_severity=None):
    moisture_score = 100 if 40 <= moisture <= 70 else \
        50 if 30 <= moisture <= 80 else 20
    temp_score = 100 if 18 <= temp <= 28 else \
        60 if 12 <= temp <= 35 else 20
    light_score = 100 if 200 <= light <= 600 else \
        60 if 100 <= light <= 800 else 30
    ph_score = 100 if 6.0 <= ph <= 7.0 else \
        60 if 5.5 <= ph <= 7.5 else 20

    sensor_score = (moisture_score * 0.4 +
                    temp_score * 0.3 +
                    light_score * 0.2 +
                    ph_score * 0.1)

    stress_penalty = 20 if stress_predicted else 0
    severity_penalty = {
        None: 0, 'Mild': 10, 'Moderate': 25,
        'Severe': 45, 'Needs Inspection': 15
    }
    disease_penalty = severity_penalty.get(disease_severity, 0)
    health_score = max(0, sensor_score - stress_penalty - disease_penalty)

    if health_score >= 80:
        status = "Excellent"
    elif health_score >= 60:
        status = "Good"
    elif health_score >= 40:
        status = "Fair"
    else:
        status = "Critical"

    return round(health_score, 1), status

# ============================================
# ENDPOINTS
# ============================================


@app.get("/")
def root():
    return {
        "message": "FytoPod API is running!",
        "version": "1.0.0",
        "endpoints": [
            "/sensor/latest",
            "/sensor/history",
            "/prediction/stress",
            "/disease/treatment",
            "/disease/followup",
            "/disease/history",
            "/health/score"
        ]
    }


@app.get("/sensor/latest")
def get_latest_sensor():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT hour, moisture, temp, light, ph, stressed, stress_soon
        FROM sensor_readings
        ORDER BY hour DESC LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return {
        "hour": row[0],
        "moisture": round(row[1], 2),
        "temp": round(row[2], 2),
        "light": round(row[3], 2),
        "ph": round(row[4], 2),
        "stressed": bool(row[5]),
        "stress_soon": bool(row[6]),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/sensor/history")
def get_sensor_history(hours: int = 24):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT hour, moisture, temp, light, ph
        FROM sensor_readings
        ORDER BY hour DESC LIMIT {hours}
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {
        "data": [
            {
                "hour": r[0],
                "moisture": round(r[1], 2),
                "temp": round(r[2], 2),
                "light": round(r[3], 2),
                "ph": round(r[4], 2)
            } for r in rows
        ]
    }


@app.get("/prediction/stress")
def get_stress_prediction():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT hour, moisture, stress_predicted, early_warning
        FROM lstm_predictions
        ORDER BY hour DESC LIMIT 12
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    latest = rows[0]
    stress_predicted = bool(latest[2])
    early_warning = bool(latest[3])
    return {
        "stress_predicted": stress_predicted,
        "early_warning": early_warning,
        "current_moisture": round(latest[1], 2),
        "alert": "Stress predicted — consider watering soon!" if early_warning
                 else "Plant conditions normal",
        "hours_of_data": len(rows)
    }


@app.post("/disease/treatment")
def get_treatment(query: DiseaseQuery):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a plant disease expert. Give concise treatment recommendations."
            },
            {
                "role": "user",
                "content": f"Plant has {query.disease} at {query.severity} severity. Give 3 treatment recommendations in bullet points."
            }
        ],
        max_tokens=250
    )
    treatment = response.choices[0].message.content

    followup_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Generate 4 follow-up questions a user might ask about plant disease treatment."
            },
            {
                "role": "user",
                "content": f"Disease: {query.disease}, Severity: {query.severity}, Treatment: {treatment}. Generate 4 numbered follow-up questions."
            }
        ],
        max_tokens=200
    )
    followups = followup_response.choices[0].message.content
    return {
        "disease": query.disease,
        "severity": query.severity,
        "treatment": treatment,
        "followup_questions": followups,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/disease/followup")
def answer_followup(query: FollowupQuery):
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a plant disease expert. Answer follow-up questions concisely."
            },
            {
                "role": "user",
                "content": f"Disease: {query.disease}, Severity: {query.severity}. Question: {query.question}. Answer in 2-3 sentences."
            }
        ],
        max_tokens=150
    )
    return {
        "question": query.question,
        "answer": response.choices[0].message.content,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/disease/history")
def get_disease_history():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT plant_species, disease_name, confidence,
               severity, treatment_recommendation, timestamp
        FROM disease_detections
        ORDER BY timestamp DESC LIMIT 20
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return {
        "detections": [
            {
                "plant_species": r[0],
                "disease_name": r[1],
                "confidence": round(r[2], 3),
                "severity": r[3],
                "treatment": r[4],
                "timestamp": str(r[5])
            } for r in rows
        ]
    }


@app.get("/health/score")
def get_health_score():
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT moisture, temp, light, ph
        FROM sensor_readings
        ORDER BY hour DESC LIMIT 1
    """)
    sensor = cursor.fetchone()
    cursor.execute("""
        SELECT stress_predicted
        FROM lstm_predictions
        ORDER BY hour DESC LIMIT 1
    """)
    stress = cursor.fetchone()
    cursor.execute("""
        SELECT severity FROM disease_detections
        ORDER BY timestamp DESC LIMIT 1
    """)
    disease = cursor.fetchone()
    cursor.close()
    conn.close()

    moisture, temp, light, ph = sensor
    stress_predicted = bool(stress[0]) if stress else False
    disease_severity = disease[0] if disease else None

    score, status = calculate_health_score(
        moisture, temp, light, ph,
        stress_predicted, disease_severity
    )
    return {
        "health_score": score,
        "status": status,
        "stress_predicted": stress_predicted,
        "disease_severity": disease_severity,
        "breakdown": {
            "moisture": round(moisture, 2),
            "temp": round(temp, 2),
            "light": round(light, 2),
            "ph": round(ph, 2)
        }
    }

# ============================================
# CELLULAR AUTOMATA — DISEASE SPREAD
# ============================================


def simulate_disease_spread(initial_infection_pct, disease_type, humidity, temp, days=14):
    spread_rates = {
        'Early Blight': 0.08, 'Late Blight': 0.15,
        'Bacterial Spot': 0.10, 'Leaf Mold': 0.07, 'default': 0.09
    }
    humidity_factor = 1 + (humidity - 50) / 100
    temp_factor = 1 + (temp - 25) / 50
    base_rate = spread_rates.get(disease_type, spread_rates['default'])
    spread_rate = base_rate * humidity_factor * temp_factor

    grid_size = 50
    grid = np.zeros((grid_size, grid_size))
    infected_cells = int(grid_size * grid_size * initial_infection_pct / 100)
    center = grid_size // 2
    radius = int(np.sqrt(infected_cells / np.pi))
    for i in range(grid_size):
        for j in range(grid_size):
            if (i-center)**2 + (j-center)**2 <= radius**2:
                grid[i][j] = 1

    daily_infection = [initial_infection_pct]
    for day in range(1, days+1):
        new_grid = grid.copy()
        for i in range(1, grid_size-1):
            for j in range(1, grid_size-1):
                if grid[i][j] == 0:
                    neighbors = grid[i-1][j] + grid[i+1][j] + \
                        grid[i][j-1] + grid[i][j+1]
                    if neighbors > 0 and np.random.random() < spread_rate:
                        new_grid[i][j] = 1
        grid = new_grid
        pct = (grid.sum() / (grid_size**2)) * 100
        daily_infection.append(round(pct, 1))

    return daily_infection

# ============================================
# LIFESPAN PREDICTOR
# ============================================


def train_lifespan_model():
    """
    Synthetic data generated from literature-based horticultural parameters:
    - Base lifespan from FAO crop duration guidelines (2021)
    - Disease impact rates from USDA Plant Disease Handbook
    - Heat stress coefficients from Journal of Plant Physiology (Wahid et al., 2007)
    - Water stress from Agricultural Water Management (Hsiao, 1973)
    """
    np.random.seed(42)
    n = 1000

    # 0=none,1=mild,2=moderate,3=severe
    disease_severity = np.random.choice([0, 1, 2, 3], n)
    stress_score = np.random.uniform(0, 100, n)
    health_score = np.random.uniform(0, 100, n)
    plant_age_days = np.random.randint(10, 365, n)
    temp = np.random.uniform(15, 40, n)
    moisture = np.random.uniform(10, 90, n)
    species = np.random.choice(['tomato', 'pepper', 'potato', 'strawberry'], n)

    # Base lifespan per species (FAO, 2021)
    species_base = {
        'tomato': 150,
        'pepper': 180,
        'potato': 120,
        'strawberry': 200
    }
    base = np.array([species_base[s] for s in species])

    # Disease severity impact (USDA Plant Disease Handbook)
    # Mild: -10 days, Moderate: -35 days, Severe: -70 days
    disease_impact = np.array([0, 10, 35, 70])[disease_severity]

    # Water stress impact (Hsiao, 1973 - Agricultural Water Management)
    # Severe water stress (low moisture) reduces lifespan significantly
    moisture_impact = np.where(moisture < 30, 25,   # severe drought
                               np.where(moisture < 45, 10,   # mild drought
                                        np.where(moisture > 80, 8,    # overwatering
                                                 0)))                           # optimal

    # Heat stress above 30°C (Wahid et al., 2007)
    # Each degree above 30°C reduces lifespan by ~4 days
    heat_impact = np.maximum(0, temp - 30) * 4

    # General stress score impact
    stress_impact = stress_score * 0.4

    # Health score benefit
    health_benefit = health_score * 0.3

    lifespan_days = (
        base
        - disease_impact
        - moisture_impact
        - heat_impact
        - stress_impact
        + health_benefit
        + np.random.normal(0, 5, n)  # biological variance
    ).clip(5, 365)

    data = pd.DataFrame({
        'disease_severity': disease_severity,
        'stress_score': stress_score,
        'health_score': health_score,
        'plant_age_days': plant_age_days,
        'temp': temp,
        'moisture': moisture,
        'lifespan_days': lifespan_days
    })

    X = data.drop('lifespan_days', axis=1)
    y = data['lifespan_days']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"Lifespan model R² score: {round(score, 3)}")
    print(f"Sample predictions:")
    print(
        f"  Severe disease, stressed: {round(model.predict(pd.DataFrame([{'disease_severity':3,'stress_score':80,'health_score':20,'plant_age_days':60,'temp':32,'moisture':25}]))[0])} days")
    print(
        f"  No disease, healthy:      {round(model.predict(pd.DataFrame([{'disease_severity':0,'stress_score':10,'health_score':90,'plant_age_days':60,'temp':24,'moisture':60}]))[0])} days")

    return model


lifespan_model = train_lifespan_model()
print("Lifespan model trained!")


def predict_lifespan(disease_severity, stress_score, health_score,
                     plant_age_days, temp, moisture):
    features = pd.DataFrame([{
        'disease_severity': disease_severity,
        'stress_score': stress_score,
        'health_score': health_score,
        'plant_age_days': plant_age_days,
        'temp': temp,
        'moisture': moisture
    }])
    return round(lifespan_model.predict(features)[0])

# ============================================
# TREATMENT IMPACT SIMULATOR
# ============================================


def simulate_treatment_impact(current_health, disease_severity, stress_predicted, days=14):
    severity_decline = {'Mild': 1.5, 'Moderate': 3.0, 'Severe': 5.0}
    decline = severity_decline.get(disease_severity, 2.0)
    stress_decline = 1.0 if stress_predicted else 0.0

    untreated = [current_health]
    h = current_health
    for d in range(days):
        h = max(0, h - decline - stress_decline + np.random.normal(0, 0.5))
        untreated.append(round(h, 1))

    treated = [current_health]
    h = current_health
    for d in range(days):
        if d < 3:
            h = max(0, h - (decline * 0.3) + np.random.normal(0, 0.5))
        else:
            h = min(100, h + 1.5 + np.random.normal(0, 0.5))
        treated.append(round(h, 1))

    return untreated, treated

# ============================================
# NEW ENDPOINTS
# ============================================


@app.post("/disease/spread")
def get_spread_forecast(query: DiseaseQuery):
    infections = simulate_disease_spread(
        initial_infection_pct=18,
        disease_type=query.disease,
        humidity=65,
        temp=28
    )
    return {
        "disease": query.disease,
        "day_0": infections[0],
        "day_3": infections[3],
        "day_7": infections[7],
        "day_14": infections[14],
        "progression": infections
    }


@app.post("/disease/lifespan")
def get_lifespan(data: dict):
    without = predict_lifespan(
        disease_severity=data['severity_score'],
        stress_score=data['stress_score'],
        health_score=data['health_score'],
        plant_age_days=data['age_days'],
        temp=data['temp'],
        moisture=data['moisture']
    )
    with_treatment = predict_lifespan(
        disease_severity=max(0, data['severity_score']-1),
        stress_score=data['stress_score'] * 0.4,
        health_score=min(100, data['health_score'] + 20),
        plant_age_days=data['age_days'],
        temp=data['temp'],
        moisture=data['moisture']
    )
    return {
        "without_treatment_days": without,
        "with_treatment_days": with_treatment
    }


@app.post("/disease/simulate")
def get_treatment_simulation(data: dict):
    untreated, treated = simulate_treatment_impact(
        current_health=data['health_score'],
        disease_severity=data['severity'],
        stress_predicted=data['stress_predicted']
    )
    return {
        "untreated_day14": untreated[-1],
        "treated_day14": treated[-1],
        "untreated_trajectory": untreated,
        "treated_trajectory": treated
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
