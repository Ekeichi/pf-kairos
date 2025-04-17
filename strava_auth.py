import os
import requests
import json
from flask import Blueprint, request, redirect, url_for, session, flash

# Configuration Strava
# Remplacez ces valeurs par vos propres identifiants Strava
STRAVA_CLIENT_ID = os.environ.get('STRAVA_CLIENT_ID', '141778')
STRAVA_CLIENT_SECRET = os.environ.get('STRAVA_CLIENT_SECRET', 'a334c280c5e9cd771d1a4659b58ce9e2cfe183f4')
STRAVA_REDIRECT_URI = os.environ.get('STRAVA_REDIRECT_URI', 'http://localhost:5002/strava/callback')
TOKENS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strava_tokens.json')

# Blueprint pour les routes Strava
strava_bp = Blueprint('strava', __name__, url_prefix='/strava')

def save_token(user_id, tokens):
    """Sauvegarde le token de l'utilisateur dans un fichier JSON"""
    all_tokens = {}
    if os.path.exists(TOKENS_FILE):
        with open(TOKENS_FILE, 'r') as f:
            try:
                all_tokens = json.load(f)
            except json.JSONDecodeError:
                all_tokens = {}
    
    all_tokens[user_id] = tokens
    with open(TOKENS_FILE, 'w') as f:
        json.dump(all_tokens, f)

def load_token(user_id):
    """Charge le token d'un utilisateur depuis le fichier JSON"""
    if not os.path.exists(TOKENS_FILE):
        return None
    
    with open(TOKENS_FILE, 'r') as f:
        try:
            all_tokens = json.load(f)
            return all_tokens.get(user_id)
        except json.JSONDecodeError:
            return None

def refresh_token(user_id):
    """Rafraîchit le token d'accès si nécessaire"""
    tokens = load_token(user_id)
    if tokens is None:
        return None
    
    # Si le token n'est pas expiré, on le retourne directement
    import time
    if tokens.get('expires_at', 0) > time.time():
        return tokens.get('access_token')
    
    # Sinon, on rafraîchit le token
    refresh_token = tokens.get('refresh_token')
    if refresh_token:
        response = requests.post('https://www.strava.com/oauth/token', data={
            'client_id': STRAVA_CLIENT_ID,
            'client_secret': STRAVA_CLIENT_SECRET,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token'
        })
        
        if response.status_code == 200:
            new_tokens = response.json()
            # Ajouter le refresh_token si absent dans la réponse
            if 'refresh_token' not in new_tokens and refresh_token:
                new_tokens['refresh_token'] = refresh_token
            
            save_token(user_id, new_tokens)
            return new_tokens.get('access_token')
    
    return None

@strava_bp.route('/login')
def login():
    """Redirige l'utilisateur vers la page d'autorisation Strava"""
    # URL d'autorisation Strava
    auth_url = f"https://www.strava.com/oauth/authorize?client_id={STRAVA_CLIENT_ID}&response_type=code&redirect_uri={STRAVA_REDIRECT_URI}&approval_prompt=force&scope=activity:read_all,profile:read_all"
    return redirect(auth_url)

@strava_bp.route('/callback')
def callback():
    """Gère le callback de Strava après autorisation"""
    # Vérifier s'il y a une erreur
    if 'error' in request.args:
        flash("Erreur lors de l'autorisation Strava.")
        return redirect(url_for('predict'))
    
    # Récupérer le code d'autorisation
    code = request.args.get('code')
    if not code:
        flash("Aucun code d'autorisation reçu de Strava.")
        return redirect(url_for('predict'))
    
    # Échanger le code contre un token d'accès
    response = requests.post('https://www.strava.com/oauth/token', data={
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code'
    })
    
    if response.status_code != 200:
        flash(f"Erreur lors de l'échange du code: {response.text}")
        return redirect(url_for('predict'))
    
    # Récupérer les informations du token
    token_data = response.json()
    athlete_data = token_data.get('athlete', {})
    athlete_id = athlete_data.get('id')
    
    if not athlete_id:
        flash("Impossible de récupérer l'ID de l'athlète.")
        return redirect(url_for('predict'))
    
    # Sauvegarder le token pour cet utilisateur
    save_token(str(athlete_id), token_data)
    
    # Stocker l'ID Strava dans la session
    session['strava_athlete_id'] = str(athlete_id)
    
    flash("Connexion Strava réussie!")
    return redirect(url_for('predict'))

@strava_bp.route('/logout')
def logout():
    """Déconnecte l'utilisateur de Strava"""
    if 'strava_athlete_id' in session:
        session.pop('strava_athlete_id')
    
    flash("Déconnexion Strava réussie!")
    return redirect(url_for('predict'))

def get_athlete_records(athlete_id):
    """Récupère les records personnels d'un athlète depuis Strava"""
    access_token = refresh_token(athlete_id)
    if not access_token:
        return None
    
    # Récupérer les informations de l'athlète
    athlete_response = requests.get('https://www.strava.com/api/v3/athlete', headers={
        'Authorization': f'Bearer {access_token}'
    })
    
    if athlete_response.status_code != 200:
        return None
    
    athlete_data = athlete_response.json()
    
    # Récupérer les activités récentes pour trouver les records
    activities_response = requests.get('https://www.strava.com/api/v3/athlete/activities', params={
        'per_page': 100  # Récupérer un maximum d'activités
    }, headers={
        'Authorization': f'Bearer {access_token}'
    })
    
    if activities_response.status_code != 200:
        return None
    
    activities = activities_response.json()
    
    # Récupérer les records de running
    running_activities = [a for a in activities if a.get('type') == 'Run']
    
    # Distances standards pour les records (en mètres)
    standard_distances = {
        '1000m': 1000,
        '1500m': 1500,
        '1 mile': 1609.34,
        '3000m': 3000,
        '5km': 5000,
        '10km': 10000,
        'semi-marathon': 21097.5,
        'marathon': 42195
    }
    
    # Dictionnaire pour stocker les meilleurs temps pour chaque distance
    best_times = {}
    
    # Parcourir les activités et trouver les meilleurs temps
    for activity in running_activities:
        # Ignorer les activités sans distance ou temps
        if not activity.get('distance') or not activity.get('moving_time'):
            continue
        
        # Pour les activités proches des distances standards
        for name, distance in standard_distances.items():
            # Tolérance de 5% pour la distance
            if abs(activity['distance'] - distance) / distance < 0.05:
                # Si c'est un meilleur temps ou si on n'a pas encore de temps pour cette distance
                if name not in best_times or activity['moving_time'] < best_times[name]['time']:
                    best_times[name] = {
                        'time': activity['moving_time'],
                        'date': activity['start_date']
                    }
        
        # Vérifier également les segments pour des distances plus courtes
        # Cette partie est plus complexe et nécessiterait des appels supplémentaires à l'API
    
    # Convertir les temps en secondes en format HH:MM:SS
    records = {}
    for distance, data in best_times.items():
        seconds = data['time']
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        
        if hours > 0:
            time_str = f"{int(hours)}:{int(minutes):02d}:{int(remaining_seconds):02d}"
        else:
            time_str = f"{int(minutes):02d}:{int(remaining_seconds):02d}"
        
        records[distance] = time_str
    
    return records
