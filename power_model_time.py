import numpy as np
from scipy.optimize import minimize
from scipy.special import lambertw

# Fonction pour convertir un temps au format hh:mm:ss en minutes
def time_to_minutes(time_str):
    """
    Convertit un temps au format 'hh:mm:ss', 'mm:ss', ou 'ss.ms' en minutes
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
        
    parts = time_str.split(':')
    
    if len(parts) == 3:  # Format hh:mm:ss
        hours, minutes, seconds = map(float, parts)
        return hours * 60 + minutes + seconds / 60
    elif len(parts) == 2:  # Format mm:ss
        minutes, seconds = map(float, parts)
        return minutes + seconds / 60
    elif len(parts) == 1:  # Format seconds.milliseconds
        return float(parts[0]) / 60
    else:
        raise ValueError("Format de temps non reconnu. Utilisez 'hh:mm:ss', 'mm:ss' ou 'ss.ms'")

# Fonction pour convertir des minutes en format hh:mm:ss
def minutes_to_time_str(minutes):
    """
    Convertit des minutes en format 'hh:mm:ss', 'mm:ss' ou 'ss.cc'
    """
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    secs = int((minutes * 60) % 60)
    centisecs = int(((minutes * 60 * 100) % 100))
    
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:02d}"
    elif mins > 0:
        return f"{mins}:{secs:02d}"
    else:
        return f"{secs}.{centisecs:02d}"

# Fonction pour calculer le temps prédit T(d) pour une distance d
def predicted_time(d, vm, tc, gamma_s, gamma_l):
    dc = vm * tc  # Distance critique
    if d <= dc:
        arg = - (1 / gamma_s) * (d / dc) * np.exp(-1 / gamma_s)
        W = lambertw(arg, k=-1).real
        T = - (d / (gamma_s * vm)) / W
    else:
        arg = - (1 / gamma_l) * (d / dc) * np.exp(-1 / gamma_l)
        W = lambertw(arg, k=-1).real
        T = - (d / (gamma_l * vm)) / W
    return T

# Fonction d'erreur à minimiser
def error_function(params, distances, real_times):
    vm, tc, gamma_s, gamma_l = params
    predicted_times = [predicted_time(d, vm, tc, gamma_s, gamma_l) for d in distances]
    error = sum(((real - pred) / real) ** 2 for real, pred in zip(real_times, predicted_times))
    return error

# ====== CONFIGURATION DES DONNÉES ======
# Distances en mètres
distances = [1000, 1609.34, 3000, 5000, 10000, 21095]

# Temps réels au format mixte (vous pouvez utiliser des nombres ou des chaînes de caractères)
real_times_str = [
    '3:29',       # 1000m
    '5:49',       # 1mi 
    '12:00',       # 3000m 
    '19:37',      # 5000m
    '44:53',      # 10000m
    '1:44:19',      # Semi-Marathon
]

# Conversion automatique des temps en minutes
real_times = [time_to_minutes(t) for t in real_times_str]

# Valeurs initiales pour les paramètres
initial_params = [400, 10, 0.1, 0.05]  # vm en m/min, tc en min, gamma_s, gamma_l

# Optimisation pour minimiser l'erreur
result = minimize(error_function, initial_params, args=(distances, real_times), method='L-BFGS-B',
                  bounds=[(100, 600), (3, 15), (0.01, 1), (0.01, 1)])

# # Résultats
# if result.success:
#     vm, tc, gamma_s, gamma_l = result.x
#     print(f"Paramètres estimés : vm = {vm:.2f} m/min, tc = {tc:.2f} min, gamma_s = {gamma_s:.4f}, gamma_l = {gamma_l:.4f}")
    
#     # Affichage des temps prédits
#     print("\nComparaison des temps réels et prédits:")
#     print("Distance | Temps réel | Temps prédit | Différence | % Erreur")
#     print("-" * 70)
#     for d, rt_str, rt in zip(distances, real_times_str, real_times):
#         pt = predicted_time(d, vm, tc, gamma_s, gamma_l)
#         diff = pt - rt
#         pct_error = (pt/rt - 1) * 100
#         print(f"{d:8.0f} | {rt_str:9s} | {minutes_to_time_str(pt):11s} | {diff:+9.2f} | {pct_error:+6.1f}%")
    
#     # Distance critique
#     dc = vm * tc
#     print(f"\nDistance critique (dc) = {dc:.2f} m")
    
#     # Vitesse maximale en différentes unités
#     print(f"Vitesse maximale (vm) = {vm:.2f} m/min = {vm/60:.2f} m/s = {vm*60/1000:.2f} km/h")
#     print(f"Allure maximale = {1000/(vm):.2f} min/km = {1609.34/(vm):.2f} min/mile")
# else:
#     print("L'optimisation n'a pas convergé. Essayez avec d'autres valeurs initiales ou plus de données.")




# t = predicted_time(160000, vm, tc, gamma_s, gamma_l)
# print(t)




