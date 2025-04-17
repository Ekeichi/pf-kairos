# Kairos 1 - Prédiction de temps de course

Kairos 1 est un outil de prédiction de temps de course qui utilise des données GPX et des modèles mathématiques pour estimer vos performances sur des parcours spécifiques. Il prend en compte le dénivelé du parcours et les conditions météorologiques pour fournir des prédictions précises.

## Fonctionnalités

- Upload de fichiers GPX pour l'analyse de parcours
- Estimation du temps de course basée sur vos records personnels
- Ajustement en fonction du dénivelé du parcours
- Intégration des données météorologiques en temps réel
- Visualisation graphique des résultats
- Interface web intuitive
- **Intégration avec Strava** pour importer automatiquement vos records personnels

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/yourusername/peakflow.git
cd peakflow
```

2. Installez les dépendances requises :
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement pour l'API Strava :
```bash
export STRAVA_CLIENT_ID=votre_client_id
export STRAVA_CLIENT_SECRET=votre_client_secret
export STRAVA_REDIRECT_URI=http://localhost:5002/strava/callback
```

4. Lancez l'application :
```bash
python app.py
```

5. Accédez à l'application dans votre navigateur à l'adresse `http://localhost:5002`.

## Utilisation

1. Sur la page d'accueil, cliquez sur "START NOW" pour accéder à la page de prédiction.
2. Vous pouvez vous connecter à Strava pour importer automatiquement vos records personnels en cliquant sur le bouton "Connect with Strava".
3. Alternativement, vous pouvez entrer manuellement vos records personnels pour obtenir une prédiction personnalisée.
4. Uploadez votre fichier GPX contenant le parcours à analyser.
5. Si vous le souhaitez, activez l'option météo pour prendre en compte les conditions météorologiques actuelles.
6. Lancez la prédiction et visualisez les résultats.
7. Téléchargez les résultats au format JSON si nécessaire.

## Structure du projet

```
peakflow/
├── app.py                 # Application Flask principale
├── web_predictor.py       # Fonctions de prédiction pour l'interface web
├── predictor_pipeline.py  # Pipeline de prédiction principal
├── power_model_time.py    # Modèle de puissance pour la prédiction du temps
├── influence_Den.py       # Calcul de l'influence du dénivelé
├── weather.py             # Intégration des données météorologiques
├── strava_auth.py         # Intégration de l'API Strava
├── static/                # Fichiers statiques (CSS, JS)
├── templates/             # Templates HTML
└── uploads/               # Dossier pour les fichiers uploadés
```

## Modèle de prédiction

Le modèle de prédiction utilise un modèle de puissance critique pour estimer le temps de base, puis applique des ajustements en fonction du dénivelé et des conditions météorologiques. La prédiction est basée sur les concepts suivants :

1. **Modèle de puissance** : Utilise les records personnels pour modéliser la capacité de performance de l'athlète.
2. **Ajustement au dénivelé** : Calcule l'impact du dénivelé sur la performance en utilisant le concept de GAP (Grade Adjusted Pace).
3. **Ajustement météo** : Tient compte de la température, de l'humidité et du vent pour une prédiction plus précise.
