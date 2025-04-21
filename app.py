import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import io
import base64

# Importer les modules existants
import predictor_pipeline
import power_model_time
import influence_Den
import weather
import web_predictor

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'gpx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB
app.secret_key = "peakflow_secret_key"


# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Vérifier si la requête contient un fichier
    if 'gpx_file' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(url_for('predict'))
    
    file = request.files['gpx_file']
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(url_for('predict'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Récupérer les records personnels
        records = {}
        
        # Option 1: Records personnels saisis manuellement
        if 'enable_custom_records' in request.form:
            for i in range(7):  # Nous aurons 7 champs de records
                distance_key = f'distance_{i}'
                time_key = f'time_{i}'
                
                if distance_key in request.form and time_key in request.form:
                    distance = request.form[distance_key].strip()
                    time = request.form[time_key].strip()
                    if distance and time:
                        records[distance] = time
        
        
        # Récupérer les données météo
        use_weather = 'use_weather' in request.form
        weather_data = None
        
        if use_weather:
            try:
                # Calculer le centroïde du GPX
                lat, lon = weather.calculate_centroid(filepath)
                api_key = os.environ.get('WEATHER_API_KEY', "Uwu4dJD0howwIrgM9BsrrQEaVhlt0KUO")
                
                if 'manual_weather' in request.form:
                    # Saisie manuelle des données météo
                    temp = float(request.form['temperature'])
                    humidity = float(request.form['humidity'])
                    wind_speed = float(request.form['wind_speed'])
                    wind_direction = float(request.form['wind_direction'])
                else:
                    # Récupérer les données météo automatiquement
                    temp, humidity, wind_speed, wind_direction = weather.get_weather_data(lat, lon, api_key)
                
                weather_data = {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
            except Exception as e:
                flash(f"Erreur lors de la récupération des données météo: {e}")
                use_weather = False
        
        # Lancer la prédiction
        try:
            results = web_predictor.main_predictor_web(
                filepath, 
                weather_data=weather_data if use_weather else None,
                personal_records=records if records else None
            )
            
            if "error" in results:
                flash(results["error"])
                return redirect(url_for('predict'))
            
            # Générer le graphique et le convertir en base64 pour l'affichage
            img_bytes = io.BytesIO()
            web_predictor.visualize_results_web(results, filepath, save_to=img_bytes)
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            return render_template('results.html', 
                               results=results, 
                               filename=filename,
                               img_base64=img_base64,
                               weather_data=weather_data)
        except Exception as e:
            flash(f"Erreur lors de la prédiction: {e}")
            return redirect(url_for('predict'))
    
    else:
        flash('Format de fichier non autorisé, seuls les fichiers .gpx sont acceptés')
        return redirect(url_for('predict'))

@app.route('/download_results/<filename>')
def download_results(filename):
    results_filename = f"results_{filename}.json"
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
    
    if os.path.exists(results_path):
        return send_file(results_path, as_attachment=True)
    else:
        flash("Le fichier de résultats n'existe pas")
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)