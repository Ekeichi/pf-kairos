import gpxpy
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import io

# Importer les modules existants
import power_model_time
import influence_Den
import weather
import predictor_pipeline

def parse_personal_records(records: Dict[str, str]) -> Tuple[List[float], List[float]]:
    """
    Convertit les records personnels en listes de distances (m) et temps (min)
    
    Args:
        records: Dictionnaire des records personnels (format: {"5km": "21:30"})
        
    Returns:
        Tuple contenant deux listes:
        - distances: Liste des distances en mètres
        - times: Liste des temps en minutes
    """
    distances = []
    real_times = []
    
    for distance_str, time_str in records.items():
        # Convertir la distance en mètres si nécessaire
        if 'm' in distance_str and not distance_str.endswith('mi'):
            # Format: "1000m", "5000 m", etc.
            distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
        elif 'km' in distance_str:
            # Format: "5km", "10 km", etc.
            distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
        elif 'mi' in distance_str:
            # Format: "1mi", "5 mi", etc.
            distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1609.34
        elif 'k' in distance_str:
            # Format: "5k", "10k", etc. (en km)
            distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
        else:
            # Supposer que c'est en mètres si pas d'unité
            distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
            
        # Convertir le temps en minutes
        time_min = power_model_time.time_to_minutes(time_str)
        
        distances.append(distance)
        real_times.append(time_min)
    
    return distances, real_times

def predict_base_time_with_records(distance_km: float, records: Dict[str, str] = None) -> Dict[str, float]:
    """
    Estimation du temps de base à partir des records personnels
    
    Args:
        distance_km: Distance totale en kilomètres
        records: Dictionnaire des records personnels (format: {"5km": "21:30"})
        
    Returns:
        Dict contenant le temps de base estimé en secondes
    """
    try:
        if records and len(records) >= 3:
            # Utiliser les records personnels
            distances, real_times = parse_personal_records(records)
        else:
            # Utiliser les données par défaut
            distances = [1000, 1609.34, 3000, 5000, 10000, 21095, 42195]
            real_times_str = ['3:29', '5:49', '12:00', '19:37', '44:53', '1:44:19', '3:45:00']
            real_times = [power_model_time.time_to_minutes(t) for t in real_times_str]
        
        # Valeurs initiales pour les paramètres
        initial_params = [400, 10, 0.1, 0.05]  # vm en m/min, tc en min, gamma_s, gamma_l
        
        # Optimisation pour minimiser l'erreur
        result = power_model_time.minimize(
            power_model_time.error_function,
            initial_params,
            args=(distances, real_times),
            method='L-BFGS-B',
            bounds=[(100, 600), (3, 15), (0.01, 1), (0.01, 1)]
        )
        
        # Extraire les paramètres optimisés
        if result.success:
            vm, tc, gamma_s, gamma_l = result.x
            # Prédire le temps pour la distance donnée (en m)
            predicted_time_min = power_model_time.predicted_time(
                distance_km * 1000, vm, tc, gamma_s, gamma_l
            )
            # Convertir en secondes
            base_time_s = predicted_time_min * 60
            
            return {
                "base_time_s": base_time_s
            }
        else:
            raise ValueError("L'optimisation du modèle de puissance n'a pas convergé.")
            
    except Exception as e:
        print(f"Erreur lors de la prédiction du temps de base: {e}")
        # En cas d'erreur, utiliser la fonction standard
        return predictor_pipeline.predict_base_time(distance_km)

def visualize_results_web(results: Dict[str, Any], gpx_file_path: str, save_to=None) -> None:
    """
    Visualisation des résultats pour le web
    
    Args:
        results: Dictionnaire contenant tous les résultats des étapes précédentes
        gpx_file_path: Chemin vers le fichier GPX
        save_to: Un objet IO pour sauvegarder l'image au lieu de l'afficher (pour le web)
    """
    gpx_data = results["gpx_data"]
    points = gpx_data["gpx_points"]
    distance_km = gpx_data["distance_km"]
    
    # Récupérer les données pour les graphiques
    distances, slopes = influence_Den.calculate_slope_profile(points)
    distances_km = np.array(distances) / 1000
    elevations = [p.elevation for p in points]
    
    # Calculer les paces ajustés à partir des résultats
    baseline_speed_kmh = (3600 / results["base_time_s"]) * distance_km
    flat_pace_min_per_km = 60 / baseline_speed_kmh  # min/km
    
    # Utiliser les calculs d'influence_Den pour les paces ajustés
    adjusted_speeds = np.array([flat_pace_min_per_km / influence_Den.get_speed_adjustment_factor(slope) 
                               for slope in slopes])
    
    # Ajouter une valeur au début pour aligner avec les distances
    adjusted_paces = np.insert(adjusted_speeds, 0, flat_pace_min_per_km)
    
    # Calculer le temps par kilomètre avec fatigue progressive
    # Récupérer le temps total estimé en minutes
    total_time_min = results["time_with_elevation_s"] / 60
    total_distance = distance_km
    
    # Facteur de fatigue: augmente avec la distance parcourue
    def fatigue_factor(km_number, total_km):
        # Commence à augmenter significativement après 60% de la course
        race_completion = km_number / total_km
        if race_completion < 0.6:
            return 1.0
        elif race_completion < 0.8:
            # Augmentation modérée
            return 1.0 + 0.03 * ((race_completion - 0.6) / 0.2)
        else:
            # Augmentation plus forte dans les derniers km
            return 1.03 + 0.07 * ((race_completion - 0.8) / 0.2)
    
    # Calculer le temps prédit pour chaque kilomètre
    km_times = []  # Temps en minutes pour chaque km
    km_numbers = []  # Numéro du kilomètre
    
    for km in range(int(distance_km)):
        # Trouver tous les indices des points dans ce km
        indices = np.where((distances_km >= km) & (distances_km < (km + 1)))[0]
        if len(indices) > 0:
            # Calculer l'allure moyenne pour ce km (min/km)
            km_pace = np.mean(adjusted_paces[indices])
            
            # Appliquer le facteur de fatigue progressive
            km_pace_with_fatigue = km_pace * fatigue_factor(km+1, total_distance)
            
            # Convertir l'allure en temps pour ce km (minutes)
            km_time = km_pace_with_fatigue
            km_times.append(km_time)
            km_numbers.append(km + 1)  # Commencer à km 1 plutôt que km 0
    
    # Calculer les temps ajustés avec météo si disponible
    weather_times = []
    if "weather_adjustment_s" in results:
        weather_factor = 1 + (results["weather_adjustment_s"] / results["time_with_elevation_s"])
        weather_times = [time * weather_factor for time in km_times]
    
    # Calculer l'allure moyenne
    avg_pace = (results["time_with_elevation_s"] / 60) / distance_km
    avg_pace_weather = 0
    if "weather_adjustment_s" in results:
        avg_pace_weather = (results["final_time_s"] / 60) / distance_km
    
    # Réduire la résolution des données pour rendre le fichier JSON plus léger
    # Pour les données d'élévation, échantillonner environ 1 point tous les 100 mètres
    sample_rate = max(1, len(distances_km) // 300)  # Limiter à environ 300 points
    
    sampled_distances = [distances_km[i] for i in range(0, len(distances_km), sample_rate)]
    sampled_elevations = [elevations[i] for i in range(0, len(elevations), sample_rate)]
    
    # Créer un dictionnaire avec les données pour Plotly
    plot_data = {
        "filename": os.path.basename(gpx_file_path),
        "distances_km": sampled_distances,
        "elevations": sampled_elevations,
        "flat_pace_min_per_km": float(flat_pace_min_per_km),
        "adjusted_paces_min_per_km": adjusted_speeds.tolist(),
        "km_numbers": km_numbers,
        "km_times": km_times,
        "weather_times": weather_times,
        "avg_pace": float(avg_pace),
        "avg_pace_weather": float(avg_pace_weather),
        "has_weather": "weather_adjustment_s" in results
    }
    
    # Sauvegarder les données pour Plotly
    plot_data_path = os.path.join(os.path.dirname(gpx_file_path), f"plot_data_{os.path.basename(gpx_file_path)}.json")
    with open(plot_data_path, 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    # Si on a besoin d'une image pour PDF ou pour compatibilité, générer avec matplotlib
    if save_to:
        # Créer une figure avec 2 sous-graphiques
        plt.figure(figsize=(14, 10))
        
        # Graphique 1: Profil d'élévation
        plt.subplot(2, 1, 1)
        plt.plot(distances_km, elevations)
        plt.xlabel("Distance (km)")
        plt.ylabel("Altitude (m)")
        plt.title("Profil d'élévation")
        plt.grid(True)
        
        # Graphique 2: Temps par kilomètre
        plt.subplot(2, 1, 2)
        
        # Créer des espaces entre les barres pour une meilleure lisibilité
        bar_width = 0.35
        
        # Créer les barres pour le temps par km avec dénivelé
        bars = plt.bar(km_numbers, km_times, width=bar_width, 
                 color='blue', alpha=0.7, label="Temps par km (avec dénivelé)")
        
        if weather_times:
            # Créer les barres pour le temps par km avec météo
            weather_bars = plt.bar(np.array(km_numbers) + bar_width, weather_times, width=bar_width,
                            color='red', alpha=0.7, label="Temps par km (avec météo)")
        
        # Ajouter une ligne horizontale pour le temps moyen par km
        plt.axhline(y=avg_pace, color='blue', linestyle='--', 
                    label=f"Temps moyen: {int(avg_pace)}'{int((avg_pace-int(avg_pace))*60):02}\" /km")
        
        if "weather_adjustment_s" in results:
            plt.axhline(y=avg_pace_weather, color='red', linestyle='--', 
                        label=f"Temps moyen avec météo: {int(avg_pace_weather)}'{int((avg_pace_weather-int(avg_pace_weather))*60):02}\" /km")
        
        # Ajouter des étiquettes de temps sur chaque barre
        for i, bar in enumerate(bars):
            height = bar.get_height()
            minutes = int(height)
            seconds = int((height - minutes) * 60)
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f"{minutes}'{seconds:02}\"",
                    ha='center', va='bottom', fontsize=9)
        
        if "weather_adjustment_s" in results:
            for i, bar in enumerate(weather_bars):
                height = bar.get_height()
                minutes = int(height)
                seconds = int((height - minutes) * 60)
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f"{minutes}'{seconds:02}\"",
                        ha='center', va='bottom', fontsize=9, color='darkred')
        
        # Configurer le graphique
        plt.xlabel("Kilomètre")
        plt.ylabel("Temps (minutes)")
        plt.title("Temps prédit par kilomètre")
        plt.xticks(np.array(km_numbers) + bar_width/2 if "weather_adjustment_s" in results else km_numbers)
        plt.xlim(0.5, max(km_numbers) + 1)
        plt.ylim(0, max(km_times) * 1.2 if "weather_adjustment_s" not in results else max(max(km_times), max(weather_times)) * 1.2)
        plt.legend()
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_to, format='png')
        plt.close()

def parse_gpx_web(gpx_file_path: str) -> Dict[str, Any]:
    """
    Version web de la fonction parse_gpx, qui gère les erreurs sans quitter le programme
    """
    try:
        # Utiliser la fonction parse_gpx du module influence_Den
        points = influence_Den.parse_gpx(gpx_file_path)
        
        # Calcul de la distance et des pentes en utilisant la fonction du module
        distances, slopes = influence_Den.calculate_slope_profile(points)
        total_distance_km = distances[-1] / 1000
        
        # Calcul du profil d'élévation par segment de 1km
        elevation_profile = []
        current_km = 0
        start_elevation = points[0].elevation
        idx = 0
        
        while current_km < int(total_distance_km):
            next_km = current_km + 1
            while idx < len(distances) and distances[idx] / 1000 < next_km:
                idx += 1
                
            if idx < len(points):
                delta_elevation = points[idx].elevation - start_elevation
                elevation_profile.append(delta_elevation)
                start_elevation = points[idx].elevation
            
            current_km = next_km
        
        # Construction du résultat
        result = {
            "distance_km": total_distance_km,
            "elevation_profile": elevation_profile,
            "gpx_points": points
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Erreur lors du parsing du fichier GPX: {e}"
        print(error_msg)
        raise ValueError(error_msg)

def adjust_for_elevation_web(gpx_data: Dict[str, Any], base_time_s: float) -> Dict[str, float]:
    """
    Version web de la fonction adjust_for_elevation, qui gère les erreurs sans quitter le programme
    """
    try:
        # Récupérer les données nécessaires
        points = gpx_data["gpx_points"]
        distance_km = gpx_data["distance_km"]
        
        # Calcul du rythme de base en km/h
        # 3600 sec/heure / (base_time_s / distance_km) = km/h
        baseline_speed_kmh = (3600 / base_time_s) * distance_km
        
        # Utiliser la fonction calculate_gnr_gap pour obtenir les temps ajustés
        results = influence_Den.calculate_gnr_gap(points, flat_pace_kmh=baseline_speed_kmh)
        
        # Extraire les résultats
        flat_time_s = results['flat_time']
        adjusted_time_s = results['adjusted_time']
        
        # Calcul du temps additionnel
        elevation_adjustment_s = adjusted_time_s - flat_time_s
        
        return {
            "elevation_adjustment_s": elevation_adjustment_s,
            "time_with_elevation_s": base_time_s + elevation_adjustment_s
        }
        
    except Exception as e:
        error_msg = f"Erreur lors de l'ajustement pour le dénivelé: {e}"
        print(error_msg)
        raise ValueError(error_msg)

def adjust_for_weather_web(gpx_file_path: str, base_pace_min_per_km: float, weather_data: Dict[str, float]) -> Dict[str, float]:
    """
    Version web de la fonction adjust_for_weather, qui gère les erreurs sans quitter le programme
    """
    try:
        # Extraire les données météo
        temp = weather_data.get("temperature", 20)
        humidity = weather_data.get("humidity", 60)
        wind_speed = weather_data.get("wind_speed", 0)
        wind_direction = weather_data.get("wind_direction", 0)
        
        # Calculer l'impact météo
        base_time, adjusted_time, penalty, distance = weather.calculate_weather_impact(
            gpx_file_path, 
            base_pace_min_per_km, 
            temp, 
            humidity, 
            wind_speed, 
            wind_direction
        )
        
        # Convertir en secondes
        base_time_s = base_time * 60
        adjusted_time_s = adjusted_time * 60
        weather_adjustment_s = (adjusted_time - base_time) * 60
        
        return {
            "weather_adjustment_s": weather_adjustment_s,
            "final_time_s": adjusted_time_s
        }
        
    except Exception as e:
        error_msg = f"Erreur lors de l'ajustement pour la météo: {e}"
        print(error_msg)
        raise ValueError(error_msg)

def main_predictor_web(
    gpx_file_path: str,
    weather_data: Dict[str, float] = None,
    personal_records: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Pipeline principal de prédiction pour le web
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        weather_data: Données météo (optionnel)
        personal_records: Dictionnaire des records personnels (optionnel)
        
    Returns:
        Dict contenant tous les résultats des étapes
    """
    results = {}
    
    try:
        # Étape 1: Chargement et parsing du fichier GPX
        gpx_data = parse_gpx_web(gpx_file_path)
        results["gpx_data"] = gpx_data
        
        # Étape 2: Estimation du temps de base
        if personal_records:
            base_time_results = predict_base_time_with_records(gpx_data["distance_km"], personal_records)
        else:
            base_time_results = predictor_pipeline.predict_base_time(gpx_data["distance_km"])
            
        base_time_s = base_time_results["base_time_s"]
        base_time_hms = influence_Den.format_time(base_time_s)
        results.update(base_time_results)
        results["base_time_formatted"] = base_time_hms
        
        # Étape 3: Ajout de l'influence du dénivelé
        elevation_results = adjust_for_elevation_web(gpx_data, base_time_s)
        elevation_adjustment_s = elevation_results["elevation_adjustment_s"]
        time_with_elevation_s = elevation_results["time_with_elevation_s"]
        results.update(elevation_results)
        results["elevation_adjustment_formatted"] = influence_Den.format_time(elevation_adjustment_s)
        results["time_with_elevation_formatted"] = influence_Den.format_time(time_with_elevation_s)
        
        # Étape 4: Ajustement météo (optionnel)
        if weather_data:
            # Calculer l'allure de base en min/km en utilisant le temps avec dénivelé
            base_pace_min_per_km = (time_with_elevation_s / 60) / gpx_data["distance_km"]
            weather_results = adjust_for_weather_web(gpx_file_path, base_pace_min_per_km, weather_data)
            weather_adjustment_s = weather_results["weather_adjustment_s"]
            final_time_s = time_with_elevation_s + weather_adjustment_s
            weather_results["final_time_s"] = final_time_s
            results.update(weather_results)
            results["weather_adjustment_formatted"] = influence_Den.format_time(weather_adjustment_s)
            results["final_time_formatted"] = influence_Den.format_time(final_time_s)
        
        # Sauvegarde des résultats au format JSON
        results_filename = f"results_{os.path.basename(gpx_file_path)}.json"
        results_path = os.path.join(os.path.dirname(gpx_file_path), results_filename)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x) if hasattr(x, '__iter__') and not isinstance(x, (dict, list)) else x)
        
        return results
        
    except Exception as e:
        error_msg = f"Erreur lors de la prédiction: {str(e)}"
        print(error_msg)
        return {"error": error_msg}