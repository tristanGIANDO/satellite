# SATELLITE

## Roadmap

### 🛰️ Phase 1 : Données et prétraitement (1-2 semaines)

Objectif : télécharger des images et générer un premier dataset filtrable.

1. Choix d’une zone d’étude

   Ex : région autour de Toulouse ou Paris.

   Petite surface (10x10 km) pour rester léger.

2. Accès aux données

   Installe sentinelsat pour Sentinel-2.

   Ou utilise Google Earth Engine (très pratique pour tester sans stockage local).

3. Extraction d’images

   Critères : cloud cover < 50%, sur une même zone, plusieurs dates.

   Sauvegarde des bandes RGB (B4, B3, B2) + SCL (Scene Classification Layer).

4. Prétraitement

   Convertir GeoTIFF -> PNG ou JPG (pour entraîner le CNN).

   Découpage en patchs (128x128 par exemple) avec un script Python (rasterio + numpy).

### ☁️ Phase 2 : Détection de nuages par CNN (2-3 semaines)

Objectif : créer un modèle qui classe les images avec ou sans nuages.

1. Création du dataset

   Patchs labellisés automatiquement grâce au masque SCL :

   9 = Nuage élevé, 8 = Nuage moyen, 3 = Ombre…

   Tu peux décider d’un seuil (% pixels nuageux) pour dire si une image est "cloudy".

2. Modèle CNN simple (baseline)

   Entrée : image RGB.

   Sortie : binaire (cloud / no cloud).

   Base : architecture type LeNet ou ResNet18 avec PyTorch.

3. Entraînement + évaluation

   Accuracy, F1-score, courbe ROC.

   Optionnel : visualisation des erreurs.

### 🧩 Phase 3 : Assemblage des images filtrées (2-3 semaines)

Objectif : reconstituer une grande image sans nuages.

1. Récupération des images “propres”

   Appliquer le modèle entraîné à de nouveaux patchs.

2. Assemblage

   Utilise rasterio.merge, ou gdal_merge.py.

   Vérifie les métadonnées (projection, résolution).

3. Sauvegarde de l’image finale

   Format : GeoTIFF ou image raster simple.

   Tu peux aussi générer une carte interactive (avec folium ou leaflet si besoin).
