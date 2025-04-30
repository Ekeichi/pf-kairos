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
import csv
from datetime import datetime

# Importer les modules existants
import predictor_pipeline
import power_model_time
import influence_Den
import weather
import web_predictor

# Configuration
UPLOAD_FOLDER = '/var/data/uploads'  # Chemin pour le disque persistant sur Render
ALLOWED_EXTENSIONS = {'gpx'}
NEWSLETTER_FILE = '/var/data/newsletter_subscribers.csv'  # Fichier newsletter dans le disque persistant

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16 MB
app.secret_key = "peakflow_secret_key"

# Créer le dossier uploads s'il n'existe pas
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Créer le fichier newsletter s'il n'existe pas
os.makedirs(os.path.dirname(NEWSLETTER_FILE), exist_ok=True)  # Créer le dossier parent si nécessaire
if not os.path.exists(NEWSLETTER_FILE):
    with open(NEWSLETTER_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['email', 'subscription_date'])

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
        app.logger.info(f"Sauvegarde du fichier à : {filepath}")  # Log pour débogage
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
            
            # Générer les données pour Plotly et conserver une image pour le PDF
            img_bytes = io.BytesIO()
            web_predictor.visualize_results_web(results, filepath, save_to=img_bytes)
            img_bytes.seek(0)
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
            
            # Les données JSON pour Plotly sont déjà sauvegardées par visualize_results_web
            
            return render_template('results.html', 
                               results=results, 
                               filename=filename,
                               img_base64=img_base64,  # Conservé pour la compatibilité 
                               weather_data=weather_data)
        except Exception as e:
            flash(f"Erreur lors de la prédiction: {e}")
            return redirect(url_for('predict'))
    
    else:
        flash('Format de fichier non autorisé, seuls les fichiers .gpx sont acceptés')
        return redirect(url_for('predict'))

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Servir les fichiers depuis le dossier uploads"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    app.logger.info(f"Tentative d'accès au fichier: {file_path}")
    
    if not os.path.exists(file_path):
        app.logger.error(f"Fichier non trouvé: {file_path}")
        return f"Fichier non trouvé: {filename}", 404
    
    app.logger.info(f"Fichier trouvé, envoi de: {file_path}")
    # Définir le type MIME explicitement pour les fichiers JSON
    mime_type = 'application/json' if file_path.endswith('.json') else None
    return send_file(file_path, mimetype=mime_type)

@app.route('/download_results/<filename>')
def download_results(filename):
    results_filename = f"results_{filename}.json"
    plot_data_filename = f"plot_data_{filename}.json"
    results_path = os.path.join(app.config['UPLOAD_FOLDER'], results_filename)
    plot_data_path = os.path.join(app.config['UPLOAD_FOLDER'], plot_data_filename)
    
    if os.path.exists(results_path) and os.path.exists(plot_data_path):
        # Charger les résultats JSON
        with open(results_path, 'r') as f:
            results = json.load(f)
        with open(plot_data_path, 'r') as f:
            plot_data = json.load(f)
        
        # Créer un PDF avec les résultats
        pdf_buffer = io.BytesIO()
        
        # Styles simples
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=10,
            alignment=1
        )
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=5
        )
        
        # Contenu
        content = []
        
        # Titre
        content.append(Paragraph("Analyse de Performance", title_style))
        content.append(Spacer(1, 5))
        
        # Temps prédit
        if "final_time_formatted" in results:
            time_text = f"Temps final prédit: {results['final_time_formatted']}"
        else:
            time_text = f"Temps prédit: {results['time_with_elevation_formatted']}"
        
        content.append(Paragraph(time_text, normal_style))
        content.append(Spacer(1, 10))
        
        # Créer le graphique
        plt.figure(figsize=(8, 4))
        plt.style.use('default')
        
        # Utiliser les données du fichier plot_data
        km_numbers = plot_data["km_numbers"]
        km_times = plot_data["km_times"]
        elevations = plot_data["elevations"]
        
        if km_numbers and km_times and elevations:
            # Créer le graphique des temps par kilomètre
            plt.bar(km_numbers, km_times, color='lightblue', alpha=0.7, label="Temps par km")
            
            # Ajouter une ligne pour le temps moyen
            avg_pace = sum(km_times) / len(km_times)
            plt.axhline(y=avg_pace, color='red', linestyle='--', 
                        label=f"Temps moyen: {int(avg_pace)}'{int((avg_pace-int(avg_pace))*60):02}\" /km")
            
            # Ajouter les temps sur les barres
            for i, (km, time) in enumerate(zip(km_numbers, km_times)):
                minutes = int(time)
                seconds = int((time - minutes) * 60)
                plt.text(km, time + 0.1, f"{minutes}'{seconds:02}\"", 
                         ha='center', va='bottom', fontsize=6)
            
            plt.xlabel("Kilomètre")
            plt.ylabel("Temps (minutes)")
            plt.title("Temps prédit par kilomètre")
            plt.legend(fontsize=8)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Sauvegarder le graphique
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Ajouter le graphique au PDF
            img = Image(img_buffer, width=450, height=200)
            content.append(img)
            content.append(Spacer(1, 10))
        
        # Créer le tableau des temps par kilomètre
        if km_numbers and km_times:
            # Définir les en-têtes du tableau
            headers = ['Km', 'Temps']
            table_data = [headers]
            
            # Ajouter les données pour chaque kilomètre
            for km, time in zip(km_numbers, km_times):
                minutes = int(time)
                seconds = int((time - minutes) * 60)
                time_str = f"{minutes}'{seconds:02}\""
                table_data.append([f"{km}", time_str])
            
            # Ajouter une ligne avec le temps total
            if "final_time_formatted" in results:
                table_data.append(['Total', results['final_time_formatted']])
            else:
                table_data.append(['Total', results['time_with_elevation_formatted']])
            
            # Créer le style du tableau
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('BACKGROUND', (0, -1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ])
            
            # Alternance de couleurs pour les lignes
            for i in range(1, len(table_data)-1):
                if i % 2 == 0:
                    table_style.add('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
            
            table = Table(table_data)
            table.setStyle(table_style)
            content.append(table)
            
            # Ajouter le copyright
            content.append(Spacer(1, 10))
            copyright_style = ParagraphStyle(
                'Copyright',
                parent=styles['Normal'],
                fontSize=8,
                alignment=1,
                textColor=colors.grey
            )
            copyright_text = "© 2024 PeakFlow Technologies. Tous droits réservés."
            content.append(Paragraph(copyright_text, copyright_style))
        
        # Construire le PDF
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=30,
            leftMargin=30,
            topMargin=30,
            bottomMargin=30
        )
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

@app.route('/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email')
    
    if not email:
        flash('Please provide an email address')
        return redirect(url_for('about'))
    
    # Vérifier si l'email existe déjà
    with open(NEWSLETTER_FILE, 'r', newline='') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        if any(row[0] == email for row in reader):
            flash('You are already subscribed to our newsletter!')
            return redirect(url_for('about'))
    
    # Ajouter le nouvel abonné
    with open(NEWSLETTER_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([email, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    flash('Thank you for subscribing to our newsletter!')
    return redirect(url_for('about'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)