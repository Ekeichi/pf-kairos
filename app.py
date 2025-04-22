import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, send_file, session
from werkzeug.utils import secure_filename
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
import tempfile

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
        # Charger les résultats JSON
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Créer un PDF avec les résultats
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        subtitle_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Contenu
        content = []
        
        # Titre
        title = Paragraph(f"Résultats pour {filename}", title_style)
        content.append(title)
        content.append(Spacer(1, 20))
        
        # Section résumé
        content.append(Paragraph("Résumé", subtitle_style))
        
        # Temps prédit
        if "final_time_formatted" in results:
            time_text = f"Temps final prédit: {results['final_time_formatted']}"
        else:
            time_text = f"Temps prédit: {results['time_with_elevation_formatted']}"
        
        content.append(Paragraph(time_text, normal_style))
        content.append(Spacer(1, 10))
        
        # Ajout des données météo si disponibles
        if "weather_adjustment_s" in results:
            weather_text = [
                "Données météo:",
                f"Température: {results.get('weather_data', {}).get('temperature', 'N/A')} °C",
                f"Humidité: {results.get('weather_data', {}).get('humidity', 'N/A')} %",
                f"Vent: {results.get('weather_data', {}).get('wind_speed', 'N/A')} km/h"
            ]
            for line in weather_text:
                content.append(Paragraph(line, normal_style))
            content.append(Spacer(1, 10))
        
        # Générer et sauvegarder le graphique
        img_bytes = io.BytesIO()
        web_predictor.visualize_results_web(results, 
                                           os.path.join(app.config['UPLOAD_FOLDER'], filename), 
                                           save_to=img_bytes)
        img_bytes.seek(0)
        
        # Ajouter le graphique au PDF
        img = Image(img_bytes, width=500, height=350)
        content.append(img)
        content.append(Spacer(1, 20))
        
        # Section des temps par kilomètre
        content.append(Paragraph("Temps par kilomètre", subtitle_style))
        content.append(Spacer(1, 10))
        
        # Extraire les données du JSON
        gpx_data = results["gpx_data"]
        distance_km = gpx_data["distance_km"]
        
        # Approche simplifiée pour créer un tableau des temps par kilomètre
        # Créer des temps approchés basés sur les données déjà calculées
        km_times = []
        km_numbers = []
        
        # Si nous avons le temps total, diviser en segments de kilomètre avec légère variation
        if "time_with_elevation_s" in results:
            total_time_s = results["final_time_s"] if "final_time_s" in results else results["time_with_elevation_s"]
            avg_pace_min_per_km = (total_time_s / 60) / distance_km
            
            # Variabilité pour rendre les temps plus réalistes (+-10%)
            import random
            
            # Fixer la graine pour des résultats cohérents
            random.seed(hash(filename))
            
            # Créer des temps par kilomètre qui s'additionnent pour donner le temps total
            remaining_distance = distance_km
            remaining_time = total_time_s / 60  # en minutes
            
            for km in range(1, int(distance_km) + 1):
                # Augmenter la variabilité vers la fin (simulation de fatigue)
                variation_range = 0.05 if km < distance_km * 0.7 else 0.15
                
                # Si c'est le dernier km, utiliser exactement le temps restant
                if km == int(distance_km):
                    km_time = remaining_time
                else:
                    # Générer un temps pour ce km avec variation
                    km_time = avg_pace_min_per_km * (1 + (random.random() * 2 - 1) * variation_range)
                    
                    # Assurer que nous ne dépassons pas le temps restant
                    km_time = min(km_time, remaining_time - (remaining_distance - 1) * 0.8 * avg_pace_min_per_km)
                    
                    remaining_time -= km_time
                    remaining_distance -= 1
                
                km_times.append(km_time)
                km_numbers.append(km)
        
        # Créer le tableau des temps par kilomètre
        table_data = [['Kilomètre', 'Temps']]
        for km, time in zip(km_numbers, km_times):
            minutes = int(time)
            seconds = int((time - minutes) * 60)
            time_str = f"{minutes}'{seconds:02}\""
            table_data.append([f"Km {km}", time_str])
        
        # Ajouter une ligne avec le temps total
        if "final_time_formatted" in results:
            table_data.append(['Total', results['final_time_formatted']])
        else:
            table_data.append(['Total', results['time_with_elevation_formatted']])
        
        # Créer le style du tableau
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.gray),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ])
        
        # Alternance de couleurs pour les lignes du tableau
        for i in range(1, len(table_data)-1):
            if i % 2 == 0:
                table_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
        
        table = Table(table_data)
        table.setStyle(table_style)
        content.append(table)
        
        # Ajouter les notes de bas de page
        content.append(Spacer(1, 30))
        footer_text = "Généré par PEAKFLOW Kairos 1 - Outil d'analyse et de prédiction de performance en course à pied"
        footer = Paragraph(footer_text, styles['Italic'])
        content.append(footer)
        
        # Construire le PDF
        doc.build(content)
        pdf_buffer.seek(0)
        
        # Renvoyer le PDF
        pdf_filename = f"peakflow_results_{filename}.pdf"
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
    else:
        flash("Le fichier de résultats n'existe pas")
        return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)