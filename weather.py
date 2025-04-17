import requests
import math
import gpxpy

def get_weather_data(lat, lon, api_key):
    """Récupère les données météo depuis Tomorrow.io."""
    url = f"https://api.tomorrow.io/v4/weather/realtime?location={lat},{lon}&fields=temperature,humidity,windSpeed,windDirection&units=metric&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    
    if "data" not in data:
        raise Exception("Erreur : " + data.get("message", "Inconnu"))
    
    temp = data["data"]["values"]["temperature"]
    humidity = data["data"]["values"]["humidity"]
    wind_speed = data["data"]["values"]["windSpeed"]
    wind_direction = data["data"]["values"]["windDirection"]
    
    return temp, humidity, wind_speed, wind_direction

def calculate_centroid(gpx_file):
    """Calcule le centroïde (point central) des coordonnées du GPX."""
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    
    lat_sum, lon_sum, count = 0, 0, 0
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                lat_sum += point.latitude
                lon_sum += point.longitude
                count += 1
    
    if count == 0:
        raise ValueError("Le fichier GPX ne contient aucun point.")
    
    return lat_sum / count, lon_sum / count

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calcule le cap (bearing) entre deux points GPS en degrés."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

def wind_effect(wind_speed, wind_direction, runner_bearing):
    """Calcule l'effet du vent selon l'angle relatif."""
    angle_diff = (wind_direction - runner_bearing + 180) % 360 - 180
    if abs(angle_diff) < 45:  # Vent de face
        return wind_speed * 0.02
    elif abs(angle_diff) > 135:  # Vent de dos
        return wind_speed * -0.01
    else:  # Vent latéral ou intermédiaire
        factor = (abs(angle_diff) - 45) / 90
        return wind_speed * (0.02 - 0.03 * factor)

def calculate_weather_impact(gpx_file, base_pace_min_per_km, temp, humidity, wind_speed, wind_direction):
    """Calcule l'impact de la météo avec un parcours GPX."""
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    
    total_distance_km = 0
    total_penalty = 0
    previous_point = None
    
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                if previous_point:
                    dist = point.distance_2d(previous_point) / 1000
                    total_distance_km += dist
                    
                    bearing = calculate_bearing(
                        previous_point.latitude, previous_point.longitude,
                        point.latitude, point.longitude
                    )
                    
                    wind_penalty = wind_effect(wind_speed, wind_direction, bearing)
                    temp_penalty = (temp - 20) * 0.015 if temp > 20 else 0
                    humidity_penalty = (humidity - 60) * 0.005 if humidity > 60 and temp > 20 else 0
                    
                    segment_penalty = temp_penalty + humidity_penalty + wind_penalty
                    total_penalty += segment_penalty * dist
                
                previous_point = point
    
    base_time = total_distance_km * base_pace_min_per_km
    avg_penalty = total_penalty / total_distance_km if total_distance_km > 0 else 0
    adjusted_time = base_time * (1 + avg_penalty)
    
    return base_time, adjusted_time, avg_penalty, total_distance_km

def main():
    # Paramètres
    gpx_file = "Afternoon_Run.gpx"
    base_pace = float(input("Rythme de base (min/km, ex. 5.0) : "))
    
    # Clé API Tomorrow.io
    api_key = "Uwu4dJD0howwIrgM9BsrrQEaVhlt0KUO"  # Remplace par ta clé
    
    try:
        # Calculer le centroïde du GPX
        lat, lon = calculate_centroid(gpx_file)
        print(f"\nCentroïde du parcours : Lat {lat:.4f}, Lon {lon:.4f}")
        
        # Récupérer la météo au centroïde
        temp, humidity, wind_speed, wind_direction = get_weather_data(lat, lon, api_key)
        print(f"Météo au centroïde :")
        print(f"Température : {temp} °C")
        print(f"Humidité : {humidity} %")
        print(f"Vitesse du vent : {wind_speed} m/s")
        print(f"Direction du vent : {wind_direction}°")
        
        # Calculer l'impact
        base_time, adjusted_time, penalty, distance = calculate_weather_impact(
            gpx_file, base_pace, temp, humidity, wind_speed, wind_direction
        )
        
        # Résultats
        print(f"\nDistance totale : {distance:.2f} km")
        print(f"Temps de base : {base_time:.1f} min")
        print(f"Temps ajusté : {adjusted_time:.1f} min")
        print(f"Impact moyen : +{penalty*100:.1f}% (soit {adjusted_time - base_time:.1f} min)")
        
    except Exception as e:
        print(f"Erreur : {e}")

if __name__ == "__main__":
    main()

