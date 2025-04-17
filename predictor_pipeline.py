import gpxpy
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Importer les modules existants
import power_model_time
import influence_Den
import weather

def parse_gpx(gpx_file_path: str) -> Dict[str, Any]:
    """
    Étape 1: Chargement et parsing du fichier GPX
    Extrait les informations de base à partir du fichier .gpx
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        
    Returns:
        Dict contenant les informations extraites:
        {
            "distance_km": distance totale en km,
            "elevation_profile": liste des élévations par segment,
            "gpx_points": liste des points (lat, lon, alt)
        }
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
        print(f"Erreur lors du parsing du fichier GPX: {e}")
        sys.exit(1)

def predict_base_time(distance_km: float) -> Dict[str, float]:
    """
    Étape 2: Estimation du temps de base à partir des records
    Prédit le temps de base (sans dénivelé ni météo) en utilisant power_model_time.py
    
    Args:
        distance_km: Distance totale en kilomètres
        personal_records: Dictionnaire des records personnels (distance en m -> temps)
        
    Returns:
        Dict contenant le temps de base estimé:
        {
            "base_time_s": temps estimé en secondes
        }
    """
    # try:
    #     # Convertir les records personnels
    #     distances = []
    #     real_times_str = []
        
    #     for distance_str, time_str in personal_records.items():
    #         # Convertir la distance en mètres si nécessaire
    #         if 'm' in distance_str and not distance_str.endswith('mi'):
    #             # Format: "1000m", "5000 m", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
    #         elif 'km' in distance_str:
    #             # Format: "5km", "10 km", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
    #         elif 'mi' in distance_str:
    #             # Format: "1mi", "5 mi", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1609.34
    #         elif 'k' in distance_str:
    #             # Format: "5k", "10k", etc. (en km)
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
    #         else:
    #             # Supposer que c'est en mètres si pas d'unité
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
                
    #         distances.append(distance)
    #         real_times_str.append(time_str)
            
        # Convertir les temps en minutes
    # Training data: distance in meters and corresponding times
    distances = [1000, 1609.34, 3000, 5000, 10000, 21095, 42195]
    real_times_str =['3:29', '5:49','12:00', '19:37','44:53','1:44:19','3:45:00']
    real_times = [power_model_time.time_to_minutes(t) for t in real_times_str]
        
        # Valeurs initiales pour les paramètres
    initial_params = [400, 10, 0.1, 0.05]  # vm en m/min, tc en min, gamma_s, gamma_l
        
        # Optimisation pour minimiser l'erreur
    # Use predefined distances (training data) for the model fitting
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
            
        # except Exception as e:
        #     print(f"Erreur lors de la prédiction du temps de base: {e}")
        #     sys.exit(1)

def adjust_for_elevation(gpx_data: Dict[str, Any], base_time_s: float) -> Dict[str, float]:
    """
    Étape 3: Ajout de l'influence du dénivelé
    Ajuste le temps de course en fonction du dénivelé du fichier GPX
    
    Args:
        gpx_data: Données GPX issues de parse_gpx()
        base_time_s: Temps de base en secondes
        
    Returns:
        Dict contenant les temps ajustés:
        {
            "elevation_adjustment_s": temps additionnel dû au dénivelé,
            "time_with_elevation_s": temps total avec dénivelé
        }
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
        print(f"Erreur lors de l'ajustement pour le dénivelé: {e}")
        sys.exit(1)

def adjust_for_weather(gpx_file_path: str, base_pace_min_per_km: float, weather_data: Dict[str, float]) -> Dict[str, float]:
    """
    Étape 4: Ajustement météo
    Prend en compte la météo à l'heure et au lieu de la course
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        base_pace_min_per_km: Allure de base en min/km
        weather_data: Données météo (température, humidité, vitesse et direction du vent)
        
    Returns:
        Dict contenant les temps ajustés:
        {
            "weather_adjustment_s": ajustement dû à la météo en secondes,
            "final_time_s": temps final ajusté en secondes
        }
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
        print(f"Erreur lors de l'ajustement pour la météo: {e}")
        sys.exit(1)

def visualize_results(results: Dict[str, Any], gpx_file_path: str) -> None:
    """
    Étape 5: Visualisation des résultats
    
    Args:
        results: Dictionnaire contenant tous les résultats des étapes précédentes
        gpx_file_path: Chemin vers le fichier GPX
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
    
    # Créer une figure avec 2 sous-graphiques
    plt.figure(figsize=(14, 10))
    
    # Graphique 1: Profil d'élévation (similaire à influence_Den.py)
    plt.subplot(2, 1, 1)
    plt.plot(distances_km, elevations)
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil d'élévation")
    plt.grid(True)
    
    # Graphique 2: Temps par kilomètre
    plt.subplot(2, 1, 2)
    
    # Calculer le temps prédit pour chaque kilomètre
    km_times = []  # Temps en minutes pour chaque km
    km_numbers = []  # Numéro du kilomètre
    
    # Utiliser les calculs d'influence_Den pour les paces ajustés
    adjusted_speeds = np.array([flat_pace_min_per_km / influence_Den.get_speed_adjustment_factor(slope) 
                               for slope in slopes])
    
    # Ajouter une valeur au début pour aligner avec les distances
    adjusted_paces = np.insert(adjusted_speeds, 0, flat_pace_min_per_km)
    
    # Calculer le temps par kilomètre
    for km in range(int(distance_km)):
        # Trouver tous les indices des points dans ce km
        indices = np.where((distances_km >= km) & (distances_km < (km + 1)))[0]
        if len(indices) > 0:
            # Calculer l'allure moyenne pour ce km (min/km)
            km_pace = np.mean(adjusted_paces[indices])
            # Convertir l'allure en temps pour ce km (minutes)
            km_time = km_pace  # Puisque l'allure est en min/km et on regarde 1 km
            km_times.append(km_time)
            km_numbers.append(km + 1)  # Commencer à km 1 plutôt que km 0
    
    # Créer des espaces entre les barres pour une meilleure lisibilité
    bar_width = 0.35
    
    # Créer les barres pour le temps par km avec dénivelé
    bars = plt.bar(km_numbers, km_times, width=bar_width, 
             color='blue', alpha=0.7, label="Temps par km (avec dénivelé)")
    
    # Calculer les temps ajustés avec météo si disponible
    if "weather_adjustment_s" in results:
        weather_factor = 1 + (results["weather_adjustment_s"] / results["time_with_elevation_s"])
        weather_times = [time * weather_factor for time in km_times]
        
        # Créer les barres pour le temps par km avec météo
        weather_bars = plt.bar(np.array(km_numbers) + bar_width, weather_times, width=bar_width,
                        color='red', alpha=0.7, label="Temps par km (avec météo)")
    
    # Ajouter une ligne horizontale pour le temps moyen par km
    avg_pace = (results["time_with_elevation_s"] / 60) / distance_km
    plt.axhline(y=avg_pace, color='blue', linestyle='--', 
                label=f"Temps moyen: {int(avg_pace)}'{int((avg_pace-int(avg_pace))*60):02}\" /km")
    
    if "weather_adjustment_s" in results:
        avg_pace_weather = (results["final_time_s"] / 60) / distance_km
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
    plt.show()

def main_predictor(
    gpx_file_path: str, 
    weather_data: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Pipeline principal de prédiction
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        personal_records: Dictionnaire des records personnels
        weather_data: Données météo (optionnel)
        
    Returns:
        Dict contenant tous les résultats des étapes
    """
    results = {}
    
    # Étape 1: Chargement et parsing du fichier GPX
    print("\n--- Étape 1: Chargement et parsing du fichier GPX ---")
    gpx_data = parse_gpx(gpx_file_path)
    print(f"Distance: {gpx_data['distance_km']:.2f} km")
    print(f"Profil d'élévation: {gpx_data['elevation_profile']}")
    results["gpx_data"] = gpx_data
    
    input("\nAppuyez sur Entrée pour continuer à l'étape 2...")
    
    # Étape 2: Estimation du temps de base
    print("\n--- Étape 2: Estimation du temps de base ---")
    base_time_results = predict_base_time(gpx_data["distance_km"])
    base_time_s = base_time_results["base_time_s"]
    base_time_hms = influence_Den.format_time(base_time_s)
    print(f"Temps de base estimé: {base_time_hms} ({base_time_s:.0f} s)")
    results.update(base_time_results)
    
    input("\nAppuyez sur Entrée pour continuer à l'étape 3...")
    
    # Étape 3: Ajout de l'influence du dénivelé
    print("\n--- Étape 3: Ajout de l'influence du dénivelé ---")
    elevation_results = adjust_for_elevation(gpx_data, base_time_s)
    elevation_adjustment_s = elevation_results["elevation_adjustment_s"]
    time_with_elevation_s = elevation_results["time_with_elevation_s"]
    print(f"Ajustement dénivelé: {influence_Den.format_time(elevation_adjustment_s)} ({elevation_adjustment_s:.0f} s)")
    print(f"Temps avec dénivelé: {influence_Den.format_time(time_with_elevation_s)} ({time_with_elevation_s:.0f} s)")
    results.update(elevation_results)
    
    # Si des données météo sont fournies, continuer avec l'étape 4
    if weather_data:
        input("\nAppuyez sur Entrée pour continuer à l'étape 4...")
        
        # Étape 4: Ajustement météo
        print("\n--- Étape 4: Ajustement météo ---")
        # Calculer l'allure de base en min/km en utilisant le temps avec dénivelé
        base_pace_min_per_km = (time_with_elevation_s / 60) / gpx_data["distance_km"]
        weather_results = adjust_for_weather(gpx_file_path, base_pace_min_per_km, weather_data)
        weather_adjustment_s = weather_results["weather_adjustment_s"]
        final_time_s = time_with_elevation_s + weather_adjustment_s
        weather_results["final_time_s"] = final_time_s
        print(f"Ajustement météo: {influence_Den.format_time(weather_adjustment_s)} ({weather_adjustment_s:.0f} s)")
        print(f"Temps final ajusté: {influence_Den.format_time(final_time_s)} ({final_time_s:.0f} s)")
        results.update(weather_results)
    
    # Étape 5: Visualisation (optionnelle)
    show_viz = input("\nVoulez-vous afficher la visualisation des résultats? (o/n): ").lower() == 'o'
    if show_viz:
        print("\n--- Étape 5: Visualisation ---")
        visualize_results(results, gpx_file_path)
    
    print("\n--- Résultats finaux ---")
    if weather_data:
        final_time = results["final_time_s"]
        print(f"Temps final prédit: {influence_Den.format_time(final_time)} ({final_time:.0f} s)")
    else:
        final_time = results["time_with_elevation_s"]
        print(f"Temps final prédit (sans météo): {influence_Den.format_time(final_time)} ({final_time:.0f} s)")
    
    return results

if __name__ == "__main__":
    # Exemple d'utilisation
    gpx_file = "la-6000d-2025-la-6d-marathon.gpx"
    
    # Collecter les records personnels
    # print("\nEntrez vos records personnels (format: distance temps, ex: 5km 20:30)")
    # print("Entrez une ligne vide pour terminer la saisie")
    
    # records = {}
    # while True:
    #     record_input = input("> ")
    #     if not record_input:
    #         break
            
    #     parts = record_input.split()
    #     if len(parts) < 2:
    #         print("Format incorrect, réessayez (ex: 5km 20:30)")
    #         continue
            
    #     distance = parts[0]
    #     time = ' '.join(parts[1:])
    #     records[distance] = time
    
    # Collecter les données météo (optionnel)
    use_weather = input("\nVoulez-vous inclure les données météo? (o/n): ").lower() == 'o'
    weather_data = None
    
    if use_weather:
        try:
            # Calculer le centroïde du GPX
            lat, lon = weather.calculate_centroid(gpx_file)
            print(f"\nCentroïde du parcours : Lat {lat:.4f}, Lon {lon:.4f}")
            
            # Demander la clé API
            print("recupération API meteo")
            api_key = "Uwu4dJD0howwIrgM9BsrrQEaVhlt0KUO"
            
            if api_key:
                # Récupérer les données météo automatiquement
                temp, humidity, wind_speed, wind_direction = weather.get_weather_data(lat, lon, api_key)
                weather_data = {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
                print(f"Température: {temp} °C, Humidité: {humidity} %")
                print(f"Vent: {wind_speed} m/s, Direction: {wind_direction}°")
            else:
                # Saisie manuelle des données météo
                print("\nEntrez les données météo manuellement:")
                temp = float(input("Température (°C): "))
                humidity = float(input("Humidité (%): "))
                wind_speed = float(input("Vitesse du vent (m/s): "))
                wind_direction = float(input("Direction du vent (°): "))
                
                weather_data = {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
        except Exception as e:
            print(f"Erreur lors de la récupération des données météo: {e}")
            print("Poursuite sans les données météo")
            use_weather = False
    
    # Lancer le pipeline
    results = main_predictor(gpx_file, weather_data if use_weather else None)
    
    # Sauvegarde optionnelle des résultats
    save_results = input("\nVoulez-vous sauvegarder les résultats? (o/n): ").lower() == 'o'
    if save_results:
        output_file = input("Nom du fichier de sortie (ex: resultats.json): ")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x) if hasattr(x, '__iter__') and not isinstance(x, (dict, list)) else x)
        print(f"Résultats sauvegardés dans {output_file}")
import gpxpy
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# Importer les modules existants
import power_model_time
import influence_Den
import weather

def parse_gpx(gpx_file_path: str) -> Dict[str, Any]:
    """
    Étape 1: Chargement et parsing du fichier GPX
    Extrait les informations de base à partir du fichier .gpx
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        
    Returns:
        Dict contenant les informations extraites:
        {
            "distance_km": distance totale en km,
            "elevation_profile": liste des élévations par segment,
            "gpx_points": liste des points (lat, lon, alt)
        }
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
        print(f"Erreur lors du parsing du fichier GPX: {e}")
        sys.exit(1)

def predict_base_time(distance_km: float) -> Dict[str, float]:
    """
    Étape 2: Estimation du temps de base à partir des records
    Prédit le temps de base (sans dénivelé ni météo) en utilisant power_model_time.py
    
    Args:
        distance_km: Distance totale en kilomètres
        personal_records: Dictionnaire des records personnels (distance en m -> temps)
        
    Returns:
        Dict contenant le temps de base estimé:
        {
            "base_time_s": temps estimé en secondes
        }
    """
    # try:
    #     # Convertir les records personnels
    #     distances = []
    #     real_times_str = []
        
    #     for distance_str, time_str in personal_records.items():
    #         # Convertir la distance en mètres si nécessaire
    #         if 'm' in distance_str and not distance_str.endswith('mi'):
    #             # Format: "1000m", "5000 m", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
    #         elif 'km' in distance_str:
    #             # Format: "5km", "10 km", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
    #         elif 'mi' in distance_str:
    #             # Format: "1mi", "5 mi", etc.
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1609.34
    #         elif 'k' in distance_str:
    #             # Format: "5k", "10k", etc. (en km)
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.')) * 1000
    #         else:
    #             # Supposer que c'est en mètres si pas d'unité
    #             distance = float(''.join(c for c in distance_str if c.isdigit() or c == '.'))
                
    #         distances.append(distance)
    #         real_times_str.append(time_str)
            
        # Convertir les temps en minutes
    # Training data: distance in meters and corresponding times
    distances = [1000, 1609.34, 3000, 5000, 10000, 21095, 42195]
    real_times_str =['3:29', '5:49','12:00', '19:37','44:53','1:44:19','3:45:00']
    real_times = [power_model_time.time_to_minutes(t) for t in real_times_str]
        
        # Valeurs initiales pour les paramètres
    initial_params = [400, 10, 0.1, 0.05]  # vm en m/min, tc en min, gamma_s, gamma_l
        
        # Optimisation pour minimiser l'erreur
    # Use predefined distances (training data) for the model fitting
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
            
        # except Exception as e:
        #     print(f"Erreur lors de la prédiction du temps de base: {e}")
        #     sys.exit(1)

def adjust_for_elevation(gpx_data: Dict[str, Any], base_time_s: float) -> Dict[str, float]:
    """
    Étape 3: Ajout de l'influence du dénivelé
    Ajuste le temps de course en fonction du dénivelé du fichier GPX
    
    Args:
        gpx_data: Données GPX issues de parse_gpx()
        base_time_s: Temps de base en secondes
        
    Returns:
        Dict contenant les temps ajustés:
        {
            "elevation_adjustment_s": temps additionnel dû au dénivelé,
            "time_with_elevation_s": temps total avec dénivelé
        }
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
        print(f"Erreur lors de l'ajustement pour le dénivelé: {e}")
        sys.exit(1)

def adjust_for_weather(gpx_file_path: str, base_pace_min_per_km: float, weather_data: Dict[str, float]) -> Dict[str, float]:
    """
    Étape 4: Ajustement météo
    Prend en compte la météo à l'heure et au lieu de la course
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        base_pace_min_per_km: Allure de base en min/km
        weather_data: Données météo (température, humidité, vitesse et direction du vent)
        
    Returns:
        Dict contenant les temps ajustés:
        {
            "weather_adjustment_s": ajustement dû à la météo en secondes,
            "final_time_s": temps final ajusté en secondes
        }
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
        print(f"Erreur lors de l'ajustement pour la météo: {e}")
        sys.exit(1)

def visualize_results(results: Dict[str, Any], gpx_file_path: str) -> None:
    """
    Étape 5: Visualisation des résultats
    
    Args:
        results: Dictionnaire contenant tous les résultats des étapes précédentes
        gpx_file_path: Chemin vers le fichier GPX
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
    
    # Créer une figure avec 2 sous-graphiques
    plt.figure(figsize=(14, 10))
    
    # Graphique 1: Profil d'élévation (similaire à influence_Den.py)
    plt.subplot(2, 1, 1)
    plt.plot(distances_km, elevations)
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil d'élévation")
    plt.grid(True)
    
    # Graphique 2: Temps par kilomètre
    plt.subplot(2, 1, 2)
    
    # Calculer le temps prédit pour chaque kilomètre
    km_times = []  # Temps en minutes pour chaque km
    km_numbers = []  # Numéro du kilomètre
    
    # Utiliser les calculs d'influence_Den pour les paces ajustés
    adjusted_speeds = np.array([flat_pace_min_per_km / influence_Den.get_speed_adjustment_factor(slope) 
                               for slope in slopes])
    
    # Ajouter une valeur au début pour aligner avec les distances
    adjusted_paces = np.insert(adjusted_speeds, 0, flat_pace_min_per_km)
    
    # Calculer le temps par kilomètre
    for km in range(int(distance_km)):
        # Trouver tous les indices des points dans ce km
        indices = np.where((distances_km >= km) & (distances_km < (km + 1)))[0]
        if len(indices) > 0:
            # Calculer l'allure moyenne pour ce km (min/km)
            km_pace = np.mean(adjusted_paces[indices])
            # Convertir l'allure en temps pour ce km (minutes)
            km_time = km_pace  # Puisque l'allure est en min/km et on regarde 1 km
            km_times.append(km_time)
            km_numbers.append(km + 1)  # Commencer à km 1 plutôt que km 0
    
    # Créer des espaces entre les barres pour une meilleure lisibilité
    bar_width = 0.35
    
    # Créer les barres pour le temps par km avec dénivelé
    bars = plt.bar(km_numbers, km_times, width=bar_width, 
             color='blue', alpha=0.7, label="Temps par km (avec dénivelé)")
    
    # Calculer les temps ajustés avec météo si disponible
    if "weather_adjustment_s" in results:
        weather_factor = 1 + (results["weather_adjustment_s"] / results["time_with_elevation_s"])
        weather_times = [time * weather_factor for time in km_times]
        
        # Créer les barres pour le temps par km avec météo
        weather_bars = plt.bar(np.array(km_numbers) + bar_width, weather_times, width=bar_width,
                        color='red', alpha=0.7, label="Temps par km (avec météo)")
    
    # Ajouter une ligne horizontale pour le temps moyen par km
    avg_pace = (results["time_with_elevation_s"] / 60) / distance_km
    plt.axhline(y=avg_pace, color='blue', linestyle='--', 
                label=f"Temps moyen: {int(avg_pace)}'{int((avg_pace-int(avg_pace))*60):02}\" /km")
    
    if "weather_adjustment_s" in results:
        avg_pace_weather = (results["final_time_s"] / 60) / distance_km
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
    plt.show()

def main_predictor(
    gpx_file_path: str, 
    weather_data: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Pipeline principal de prédiction
    
    Args:
        gpx_file_path: Chemin vers le fichier GPX
        personal_records: Dictionnaire des records personnels
        weather_data: Données météo (optionnel)
        
    Returns:
        Dict contenant tous les résultats des étapes
    """
    results = {}
    
    # Étape 1: Chargement et parsing du fichier GPX
    print("\n--- Étape 1: Chargement et parsing du fichier GPX ---")
    gpx_data = parse_gpx(gpx_file_path)
    print(f"Distance: {gpx_data['distance_km']:.2f} km")
    print(f"Profil d'élévation: {gpx_data['elevation_profile']}")
    results["gpx_data"] = gpx_data
    
    input("\nAppuyez sur Entrée pour continuer à l'étape 2...")
    
    # Étape 2: Estimation du temps de base
    print("\n--- Étape 2: Estimation du temps de base ---")
    base_time_results = predict_base_time(gpx_data["distance_km"])
    base_time_s = base_time_results["base_time_s"]
    base_time_hms = influence_Den.format_time(base_time_s)
    print(f"Temps de base estimé: {base_time_hms} ({base_time_s:.0f} s)")
    results.update(base_time_results)
    
    input("\nAppuyez sur Entrée pour continuer à l'étape 3...")
    
    # Étape 3: Ajout de l'influence du dénivelé
    print("\n--- Étape 3: Ajout de l'influence du dénivelé ---")
    elevation_results = adjust_for_elevation(gpx_data, base_time_s)
    elevation_adjustment_s = elevation_results["elevation_adjustment_s"]
    time_with_elevation_s = elevation_results["time_with_elevation_s"]
    print(f"Ajustement dénivelé: {influence_Den.format_time(elevation_adjustment_s)} ({elevation_adjustment_s:.0f} s)")
    print(f"Temps avec dénivelé: {influence_Den.format_time(time_with_elevation_s)} ({time_with_elevation_s:.0f} s)")
    results.update(elevation_results)
    
    # Si des données météo sont fournies, continuer avec l'étape 4
    if weather_data:
        input("\nAppuyez sur Entrée pour continuer à l'étape 4...")
        
        # Étape 4: Ajustement météo
        print("\n--- Étape 4: Ajustement météo ---")
        # Calculer l'allure de base en min/km en utilisant le temps avec dénivelé
        base_pace_min_per_km = (time_with_elevation_s / 60) / gpx_data["distance_km"]
        weather_results = adjust_for_weather(gpx_file_path, base_pace_min_per_km, weather_data)
        weather_adjustment_s = weather_results["weather_adjustment_s"]
        final_time_s = time_with_elevation_s + weather_adjustment_s
        weather_results["final_time_s"] = final_time_s
        print(f"Ajustement météo: {influence_Den.format_time(weather_adjustment_s)} ({weather_adjustment_s:.0f} s)")
        print(f"Temps final ajusté: {influence_Den.format_time(final_time_s)} ({final_time_s:.0f} s)")
        results.update(weather_results)
    
    # Étape 5: Visualisation (optionnelle)
    show_viz = input("\nVoulez-vous afficher la visualisation des résultats? (o/n): ").lower() == 'o'
    if show_viz:
        print("\n--- Étape 5: Visualisation ---")
        visualize_results(results, gpx_file_path)
    
    print("\n--- Résultats finaux ---")
    if weather_data:
        final_time = results["final_time_s"]
        print(f"Temps final prédit: {influence_Den.format_time(final_time)} ({final_time:.0f} s)")
    else:
        final_time = results["time_with_elevation_s"]
        print(f"Temps final prédit (sans météo): {influence_Den.format_time(final_time)} ({final_time:.0f} s)")
    
    return results

if __name__ == "__main__":
    # Exemple d'utilisation
    gpx_file = "la-6000d-2025-la-6d-marathon.gpx"
    
    # Collecter les records personnels
    # print("\nEntrez vos records personnels (format: distance temps, ex: 5km 20:30)")
    # print("Entrez une ligne vide pour terminer la saisie")
    
    # records = {}
    # while True:
    #     record_input = input("> ")
    #     if not record_input:
    #         break
            
    #     parts = record_input.split()
    #     if len(parts) < 2:
    #         print("Format incorrect, réessayez (ex: 5km 20:30)")
    #         continue
            
    #     distance = parts[0]
    #     time = ' '.join(parts[1:])
    #     records[distance] = time
    
    # Collecter les données météo (optionnel)
    use_weather = input("\nVoulez-vous inclure les données météo? (o/n): ").lower() == 'o'
    weather_data = None
    
    if use_weather:
        try:
            # Calculer le centroïde du GPX
            lat, lon = weather.calculate_centroid(gpx_file)
            print(f"\nCentroïde du parcours : Lat {lat:.4f}, Lon {lon:.4f}")
            
            # Demander la clé API
            print("recupération API meteo")
            api_key = "Uwu4dJD0howwIrgM9BsrrQEaVhlt0KUO"
            
            if api_key:
                # Récupérer les données météo automatiquement
                temp, humidity, wind_speed, wind_direction = weather.get_weather_data(lat, lon, api_key)
                weather_data = {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
                print(f"Température: {temp} °C, Humidité: {humidity} %")
                print(f"Vent: {wind_speed} m/s, Direction: {wind_direction}°")
            else:
                # Saisie manuelle des données météo
                print("\nEntrez les données météo manuellement:")
                temp = float(input("Température (°C): "))
                humidity = float(input("Humidité (%): "))
                wind_speed = float(input("Vitesse du vent (m/s): "))
                wind_direction = float(input("Direction du vent (°): "))
                
                weather_data = {
                    "temperature": temp,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "wind_direction": wind_direction
                }
        except Exception as e:
            print(f"Erreur lors de la récupération des données météo: {e}")
            print("Poursuite sans les données météo")
            use_weather = False
    
    # Lancer le pipeline
    results = main_predictor(gpx_file, weather_data if use_weather else None)
    
    # Sauvegarde optionnelle des résultats
    save_results = input("\nVoulez-vous sauvegarder les résultats? (o/n): ").lower() == 'o'
    if save_results:
        output_file = input("Nom du fichier de sortie (ex: resultats.json): ")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: str(x) if hasattr(x, '__iter__') and not isinstance(x, (dict, list)) else x)
        print(f"Résultats sauvegardés dans {output_file}")
