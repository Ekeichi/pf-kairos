import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

# Fonction pour calculer le GAP
def calculate_statistical_gap(real_speed_kmh, gradient_percent):
    if gradient_percent > 0:  # Montée
        adjustment_factor = 1 + (gradient_percent * 0.10)
        gap_speed = real_speed_kmh * adjustment_factor
    elif gradient_percent < 0:  # Descente
        adjustment_factor = 1 / (1 + (abs(gradient_percent) * 0.05))
        gap_speed = real_speed_kmh * adjustment_factor
    else:  # Plat
        gap_speed = real_speed_kmh
    return np.clip(gap_speed, 4, 16)

# Calculer la pente moyenne à partir de elevation_data
def calculate_average_gradient(elevation_json):
    try:
        data = json.loads(elevation_json)
        distances = data.get('distance', [])  # en km
        elevations = data.get('altitude', [])  # en m
        if len(distances) < 2 or len(elevations) < 2:
            return 0

        # Calculer les pentes entre points consécutifs
        gradients = []
        for i in range(1, len(distances)):
            dist_km = distances[i] - distances[i-1]  # Différence en km
            elev_m = elevations[i] - elevations[i-1]  # Différence en m
            if dist_km > 0:
                gradient = (elev_m / (dist_km * 1000)) * 100  # Pente en %
                gradients.append(gradient)

        # Moyenne des pentes
        if gradients:
            return np.mean(gradients)
        return 0
    except:
        return 0

# Charger et traiter le CSV
def process_run_csv(file_path):
    data = pd.read_csv(file_path)
    print(f"Colonnes disponibles dans {file_path} :")
    print(data.columns)

    # Filtrer pour ne garder que les courses à pied
    data = data[data['type'] == 'Run'].copy()
    if data.empty:
        print("Aucune activité de type 'Run' trouvée.")
        return None

    try:
        # Convertir les colonnes en types numériques
        data['distance'] = pd.to_numeric(data['distance'], errors='coerce')  # m
        data['moving_time'] = pd.to_numeric(data['moving_time'], errors='coerce')  # s
        data['average_speed'] = pd.to_numeric(data['average_speed'], errors='coerce')  # m/s
        data['average_heartrate'] = pd.to_numeric(data['average_heartrate'], errors='coerce')  # bpm

        # Vérifier les unités de average_speed
        data['speed_kmh'] = data['average_speed'] * 3.6  # m/s -> km/h

        # Calculer une vitesse alternative pour vérification
        data['speed_kmh_alt'] = (data['distance'] / 1000) / (data['moving_time'] / 3600)  # km/h

        # Fréquence cardiaque
        data['heart_rate'] = data['average_heartrate']
        # Calculer la pente moyenne
        data['gradient_percent'] = data['elevation_data'].apply(calculate_average_gradient)
        # Calculer le GAP
        data['gap_speed_kmh'] = [calculate_statistical_gap(s, g) 
                                for s, g in zip(data['speed_kmh'], data['gradient_percent'])]
    except KeyError as e:
        print(f"Erreur : La colonne {e} n'existe pas.")
        return None
    except Exception as e:
        print(f"Erreur lors du traitement : {e}")
        return None

    return data

# Visualisation
def plot_run(data, file_name):
    plt.figure(figsize=(12, 6))
    plt.scatter(data.index, data['speed_kmh'], label="Vitesse réelle (km/h)", alpha=0.7)
    plt.scatter(data.index, data['gap_speed_kmh'], label="GAP (km/h)", alpha=0.7)
    plt.xlabel("Activité")
    plt.ylabel("Vitesse (km/h)")
    plt.title(f"Vitesse réelle vs GAP - {file_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

# Tester
if __name__ == "__main__":
    file_path = 'activities_RomainB.csv'
    run_data = process_run_csv(file_path)
    
    if run_data is not None:
        print("\nÉchantillon des données traitées :")
        print(run_data[['speed_kmh', 'speed_kmh_alt', 'gradient_percent', 'gap_speed_kmh', 'heart_rate']].head(10))
        plot_run(run_data, file_path.split('/')[-1])
        run_data.to_csv(f"processed_{file_path.split('/')[-1]}", index=False)
        print(f"Résultats sauvegardés dans 'processed_{file_path.split('/')[-1]}'")