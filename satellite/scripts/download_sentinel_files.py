import requests
from pathlib import Path
from datetime import date, timedelta

# Liste des dates et tuiles Sentinel-2 de la région parisienne
# Générer toutes les dates du mois de mai 2025
start_date = date(2025, 5, 1)
end_date = date(2025, 5, 31)
dates = [(start_date + timedelta(days=i)).isoformat() for i in range((end_date - start_date).days + 1)]
tiles = ["31UDQ"]

bands = {"red": "B04.jp2", "green": "B03.jp2", "blue": "B02.jp2", "nir": "B08.jp2"}


# Fonction pour construire l'URL Sentinel-2 AWS
def build_url(tile_code: str, date: str, band_file: str) -> str:
    utm_zone = tile_code[:2]
    lat_band = tile_code[2]
    grid_square = tile_code[3:]
    year, month, day = date.split("-")
    return f"https://sentinel-s2-l1c.s3.amazonaws.com/tiles/{utm_zone}/{lat_band}/{grid_square}/{year}/{int(month)}/{int(day)}/0/{band_file}"


# Dossier racine
root_dir = Path(r"C:\Users\giand\OneDrive\Documents\__packages__\_perso\satellite_data\sentinel2-31UDQ")

for date in dates:
    for tile in tiles:
        for band, band_file in bands.items():
            url = build_url(tile, date, band_file)
            output_dir = root_dir / date / tile / band
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / band_file

            if output_path.exists():
                print(f"[✔] {output_path.name} déjà présent.")
                continue

            print(f"[↓] {output_path.name} ({tile} - {date})...")
            try:
                response = requests.get(url, stream=True, timeout=20)
                response.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[✓] Téléchargé : {output_path}")
            except Exception as e:
                print(f"[✗] Erreur : {url}\n  → {e}")
