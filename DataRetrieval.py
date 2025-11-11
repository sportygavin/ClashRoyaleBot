import requests
import json

url = "https://royaleapi.github.io/cr-api-data/json/cards.json"
response = requests.get(url)
cards_data = response.json()

with open('data/cards_database.json', 'w') as f:
    json.dump(cards_data, f, indent=2)

print(f"✓ Downloaded {len(cards_data)} cards")
