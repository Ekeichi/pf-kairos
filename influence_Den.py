import gpxpy
import gpxpy.gpx
from math import radians, cos, sin, asin, sqrt
from collections import namedtuple
from typing import List
import matplotlib.pyplot as plt
import numpy as np

# Configuration des paramètres
GPX_FILE = 'Afternoon_Run.gpx'
BASELINE_SPEED_KMH = 10  # Vitesse de référence sur terrain plat (par défaut)
SLOPE_TOLERANCE = 2      # Tolérance de variation de pente en %
MAX_DOWNHILL_BONUS = 13  # Bonus maximal pour descente modérée en %

# Définition de la structure des points
TrackPoint = namedtuple('TrackPoint', ['latitude', 'longitude', 'elevation'])

def parse_gpx(file_path: str) -> List[TrackPoint]:
    """Lit et analyse un fichier GPX pour extraire les points de trajectoire."""
    try:
        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        points = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    track_point = TrackPoint(
                        latitude=point.latitude,
                        longitude=point.longitude,
                        elevation=point.elevation or 0  # Gestion des altitudes manquantes
                    )
                    points.append(track_point)

        if not points:
            raise ValueError("Aucun point trouvé dans le fichier GPX")

        print("First 5 points:", points[:5])  # Print the first 5 points

        return points

    except FileNotFoundError:
        print(f"Erreur: Fichier {file_path} introuvable")
        return load_example_points()

    except Exception as e:
        print(f"Erreur lors de la lecture du GPX: {e}")
        return load_example_points()

def load_example_points() -> List[TrackPoint]:
    """Charge des points d'exemple pour le débogage"""
    return [
        TrackPoint(48.8566, 2.3522, 100),
        TrackPoint(48.8567, 2.3523, 101),
        TrackPoint(48.8568, 2.3524, 102)
    ]

def haversine_distance(point1: TrackPoint, point2: TrackPoint) -> float:
    """Calcule la distance en mètres entre deux points géographiques"""
    lon1, lat1 = radians(point1.longitude), radians(point1.latitude)
    lon2, lat2 = radians(point2.longitude), radians(point2.latitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * 1000 * asin(sqrt(a))  # Rayon terrestre en mètres

def calculate_slope_profile(points: List[TrackPoint]) -> tuple:
    """Calcule le profil de pente et les distances cumulées"""
    distances = [0.0]
    slopes = []

    for i in range(len(points)-1):
        dist = haversine_distance(points[i], points[i+1])
        elevation_change = points[i+1].elevation - points[i].elevation
        slope = (elevation_change / dist) * 100 if dist > 0 else 0

        distances.append(distances[-1] + dist)
        slopes.append(slope)

    return distances, slopes

def create_slope_segments(slopes: List[float]) -> List[List[int]]:
    """Segmente le parcours en sections de pente homogène"""
    if not slopes:
        return []

    segments = []
    current_segment = [0]

    for i in range(1, len(slopes)):
        if abs(slopes[i] - slopes[current_segment[-1]]) > SLOPE_TOLERANCE:
            segments.append(current_segment)
            current_segment = [i]
        else:
            current_segment.append(i)

    segments.append(current_segment)
    return segments

def get_speed_adjustment_factor(slope: float) -> float:
    """Estimates speed adjustment factor based on slope."""
    if slope > 0:  # Uphill
        return 1 - (slope / 100) * (0.215 + 0.0057 * slope)
    elif slope < 0:  # Downhill
        abs_slope = abs(slope)
        if abs_slope <= 10:
            return 1 + (min(abs_slope, MAX_DOWNHILL_BONUS) / 100) * 0.13
        elif abs_slope > 10:
            return 1 + (MAX_DOWNHILL_BONUS / 100) * 0.13 - ((abs_slope - 10) / 100) * 0.05
    else:  # Flat
        return 1.0

def calculate_gnr_gap(points: List[TrackPoint], flat_pace_kmh: float = BASELINE_SPEED_KMH) -> dict:
    """Calcule l'indice GNR-GAP selon la méthodologie Go&Race"""
    distances, slopes = calculate_slope_profile(points)
    segments = create_slope_segments(slopes)

    # Calcul des facteurs d'ajustement
    speed_factors = []
    for segment in segments:
        avg_slope = sum(slopes[i] for i in segment) / len(segment)

        if avg_slope > 0:  # Uphill
            factor = 1 - (avg_slope / 100) * (0.215 + 0.0057 * avg_slope)
            factor = max(factor, 0.01) # Ensure factor doesn't go too low
        elif avg_slope < 0:  # Downhill
            abs_slope = abs(avg_slope)
            if abs_slope <= 10:
                factor = 1 + (min(abs_slope, MAX_DOWNHILL_BONUS) / 100) * 0.13
            elif abs_slope > 10:
                factor = 1 + (MAX_DOWNHILL_BONUS / 100) * 0.13 - ((abs_slope - 10) / 100) * 0.05
                factor = max(factor, 0.85) # Example lower bound
        else:  # Flat
            factor = 1.0

        speed_factors.append(factor)

    # Calcul des temps par segment
    baseline_speed_ms = flat_pace_kmh * 1000 / 3600
    total_time = 0.0
    start_idx = 0

    for i, segment in enumerate(segments):
        end_idx = segment[-1] + 1
        segment_dist = distances[end_idx] - distances[start_idx]
        segment_time = segment_dist / (baseline_speed_ms * speed_factors[i])
        total_time += segment_time
        start_idx = end_idx

    # Calcul de l'indice GNR-GAP
    total_distance_m = distances[-1]
    flat_time = total_distance_m / baseline_speed_ms
    gnr_gap = ((total_time - flat_time) / flat_time) * 100

    return {
        'total_distance_km': total_distance_m / 1000,
        'adjusted_time': total_time,
        'flat_time': flat_time,
        'gnr_gap': gnr_gap,
        'speed_factors': speed_factors,
        'segments': segments
    }

def format_time(seconds: float) -> str:
    """Formate un temps en secondes en HH:MM:SS"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

def format_pace(seconds_per_km: float) -> str:
    """Formate un pace en secondes par km en MM'SS/km"""
    minutes = int(seconds_per_km // 60)
    seconds = int(seconds_per_km % 60)
    return f"{minutes:02}'{seconds:02}\"/km"

# Exécution principale
if __name__ == '__main__':
    # Chargement des données
    points = parse_gpx(GPX_FILE)

    # Définir la vitesse de référence en km/h (correspondant à 5'00"/km)
    runner_flat_pace_kmh = 12  # 5 min/km = 1/5 km/min = 60/5 km/h = 12 km/h
    flat_speed_ms = runner_flat_pace_kmh * 1000 / 3600
    flat_pace_seconds_per_km = 60 / (runner_flat_pace_kmh / 60)

    # Calcul du profil de pente et des distances
    distances, slopes = calculate_slope_profile(points)
    distances_km = np.array(distances) / 1000

    # Calcul du pace adjustment factor point par point
    pace_adjustment_factors = []
    for slope in slopes:
        adjustment_factor = get_speed_adjustment_factor(slope)
        pace_adjustment_factor = 1 / adjustment_factor if adjustment_factor != 0 else np.nan
        pace_adjustment_factors.append(pace_adjustment_factor)

    # Insert a value for the first point (assuming no adjustment initially)
    pace_adjustment_factors.insert(0, 1.0)
    pace_adjustment_factors = np.array(pace_adjustment_factors)

    # Génération du graphique de profil d'élévation
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    elevations = [p.elevation for p in points]
    plt.plot(distances_km, elevations)
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (m)")
    plt.title("Profil d'élévation")
    plt.grid(True)

    # Génération du graphique de pace ajusté (en min/km)
    plt.subplot(3, 1, 2)
    flat_pace_min_per_km = 60 / runner_flat_pace_kmh
    adjusted_speeds = np.array([flat_speed_ms * get_speed_adjustment_factor(slope) for slope in slopes])
    adjusted_paces_min_per_km = 60 / (adjusted_speeds * 3.6)
    distances_km_pace = distances_km[:-1]
    plt.plot(distances_km_pace, np.full_like(distances_km_pace, flat_pace_min_per_km), label='Pace sur plat')
    plt.plot(distances_km_pace, adjusted_paces_min_per_km, label='Pace ajusté (GNR-GAP)')
    plt.xlabel("Distance (km)")
    plt.ylabel("Pace (min/km)")
    plt.title("Pace ajusté en fonction de la distance (min/km)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 15) # Set a reasonable y-limit for pace (optional)

    # Génération du graphique de facteur d'ajustement du pace
    plt.subplot(3, 1, 3)
    plt.plot(distances_km, pace_adjustment_factors, label='Facteur d\'ajustement du pace')
    plt.xlabel("Distance (km)")
    plt.ylabel("Facteur d'ajustement")
    plt.title("Facteur d'ajustement du pace en fonction de la distance")
    plt.axhline(y=1.0, color='r', linestyle='-', label='Aucun ajustement (plat)')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.8, 4.5) # Set y-limit based on the image

    plt.tight_layout()
    plt.show()

    # Calcul des indicateurs (text output)
    results = calculate_gnr_gap(points, flat_pace_kmh=runner_flat_pace_kmh)

    # Format the output string
    gnr_gap_percentage = results['gnr_gap']
    flat_time_seconds = results['flat_time']
    adjusted_time_seconds = results['adjusted_time']
    total_distance_km = results['total_distance_km']

    flat_pace_seconds_per_km = flat_time_seconds / total_distance_km
    adjusted_pace_seconds_per_km = adjusted_time_seconds / total_distance_km
    pace_difference_seconds_per_km = adjusted_pace_seconds_per_km - flat_pace_seconds_per_km

    flat_pace_formatted = format_pace(flat_pace_seconds_per_km)
    adjusted_pace_formatted = format_pace(adjusted_pace_seconds_per_km)
    pace_difference_formatted = format_pace(pace_difference_seconds_per_km)

    flat_time_formatted = format_time(flat_time_seconds)
    adjusted_time_formatted = format_time(adjusted_time_seconds)

    output_string = f"""Based on the elevation profile analysis, running on this course is estimated to result in a pace that is {gnr_gap_percentage:.1f}% {'slower' if gnr_gap_percentage > 0 else 'faster'}.

For example, a runner who maintains a pace of {flat_pace_formatted} on a flat {total_distance_km:.2f} km course, corresponding to a finish time of {flat_time_formatted}, is estimated to complete this course in {adjusted_time_formatted}. The recalculated average pace (GAP) improves to {adjusted_pace_formatted}, making it {format_pace(abs(pace_difference_seconds_per_km))} {'slower' if pace_difference_seconds_per_km > 0 else 'faster'}.

GNR-GAP index: {gnr_gap_percentage:.0f}
"""
    print(output_string)
