#!/usr/bin/env python3
"""
Clash Royale card database manager.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ClashRoyaleCardDatabase:
    def __init__(self, database_file="database/clash_royale_cards.json"):
        self.database_file = Path(database_file)
        self.data = self._load_database()
    
    def _load_database(self) -> Dict:
        """Load the card database from JSON file."""
        if not self.database_file.exists():
            raise FileNotFoundError(f"Database file not found: {self.database_file}")
        
        with open(self.database_file, 'r') as f:
            return json.load(f)
    
    def get_card_by_name(self, name: str) -> Optional[Dict]:
        """Get card information by name."""
        # Try exact match first
        for card_id, card_data in self.data['cards'].items():
            if card_data['name'].lower() == name.lower():
                return card_data
        
        # Try partial match
        for card_id, card_data in self.data['cards'].items():
            if name.lower() in card_data['name'].lower():
                return card_data
        
        return None
    
    def get_cards_by_elixir_cost(self, elixir_cost: int) -> List[Dict]:
        """Get all cards with a specific elixir cost."""
        cards = []
        for card_id, card_data in self.data['cards'].items():
            if card_data['elixir_cost'] == elixir_cost:
                cards.append(card_data)
        return cards
    
    def get_cards_by_rarity(self, rarity: str) -> List[Dict]:
        """Get all cards of a specific rarity."""
        cards = []
        for card_id, card_data in self.data['cards'].items():
            if card_data['rarity'].lower() == rarity.lower():
                cards.append(card_data)
        return cards
    
    def get_cards_by_type(self, card_type: str) -> List[Dict]:
        """Get all cards of a specific type."""
        cards = []
        for card_id, card_data in self.data['cards'].items():
            if card_data['type'].lower() == card_type.lower():
                cards.append(card_data)
        return cards
    
    def get_cards_by_arena(self, arena: str) -> List[Dict]:
        """Get all cards from a specific arena."""
        cards = []
        for card_id, card_data in self.data['cards'].items():
            if card_data['arena'].lower() == arena.lower():
                cards.append(card_data)
        return cards
    
    def search_cards(self, query: str) -> List[Dict]:
        """Search cards by name, description, or other fields."""
        query = query.lower()
        results = []
        
        for card_id, card_data in self.data['cards'].items():
            # Search in name
            if query in card_data['name'].lower():
                results.append(card_data)
                continue
            
            # Search in description
            if query in card_data['description'].lower():
                results.append(card_data)
                continue
            
            # Search in type
            if query in card_data['type'].lower():
                results.append(card_data)
                continue
        
        return results
    
    def get_elixir_cost_distribution(self) -> Dict[int, int]:
        """Get distribution of cards by elixir cost."""
        distribution = {}
        for card_id, card_data in self.data['cards'].items():
            cost = card_data['elixir_cost']
            distribution[cost] = distribution.get(cost, 0) + 1
        return distribution
    
    def get_rarity_distribution(self) -> Dict[str, int]:
        """Get distribution of cards by rarity."""
        distribution = {}
        for card_id, card_data in self.data['cards'].items():
            rarity = card_data['rarity']
            distribution[rarity] = distribution.get(rarity, 0) + 1
        return distribution
    
    def get_type_distribution(self) -> Dict[str, int]:
        """Get distribution of cards by type."""
        distribution = {}
        for card_id, card_data in self.data['cards'].items():
            card_type = card_data['type']
            distribution[card_type] = distribution.get(card_type, 0) + 1
        return distribution
    
    def get_all_cards(self) -> List[Dict]:
        """Get all cards in the database."""
        return list(self.data['cards'].values())
    
    def get_card_count(self) -> int:
        """Get total number of cards in the database."""
        return len(self.data['cards'])
    
    def get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        return {
            "total_cards": self.get_card_count(),
            "elixir_distribution": self.get_elixir_cost_distribution(),
            "rarity_distribution": self.get_rarity_distribution(),
            "type_distribution": self.get_type_distribution(),
            "metadata": self.data.get('metadata', {})
        }
    
    def find_similar_cards(self, elixir_cost: int, rarity: str = None, card_type: str = None) -> List[Dict]:
        """Find cards similar to given criteria."""
        similar_cards = []
        
        for card_id, card_data in self.data['cards'].items():
            # Must match elixir cost
            if card_data['elixir_cost'] != elixir_cost:
                continue
            
            # Optional rarity filter
            if rarity and card_data['rarity'].lower() != rarity.lower():
                continue
            
            # Optional type filter
            if card_type and card_data['type'].lower() != card_type.lower():
                continue
            
            similar_cards.append(card_data)
        
        return similar_cards
    
    def get_deck_suggestions(self, current_cards: List[str], max_elixir: float = 4.0) -> List[Dict]:
        """Get deck suggestions based on current cards."""
        # This is a simplified version - in practice, you'd want more sophisticated deck building logic
        suggestions = []
        
        # Get elixir costs of current cards
        current_costs = []
        for card_name in current_cards:
            card = self.get_card_by_name(card_name)
            if card:
                current_costs.append(card['elixir_cost'])
        
        # Calculate average elixir cost
        if current_costs:
            avg_elixir = sum(current_costs) / len(current_costs)
        else:
            avg_elixir = 0
        
        # Find cards that would fit well in the deck
        for card_id, card_data in self.data['cards'].items():
            if card_data['name'] not in current_cards:
                # Simple scoring based on elixir cost balance
                if abs(card_data['elixir_cost'] - avg_elixir) <= 1.0:
                    suggestions.append(card_data)
        
        return suggestions[:10]  # Return top 10 suggestions

def main():
    """Test the card database functionality."""
    print("=== Clash Royale Card Database Test ===")
    
    # Initialize database
    db = ClashRoyaleCardDatabase()
    
    # Test basic functionality
    print(f"Total cards in database: {db.get_card_count()}")
    
    # Test elixir cost distribution
    print("\nElixir Cost Distribution:")
    elixir_dist = db.get_elixir_cost_distribution()
    for cost in sorted(elixir_dist.keys()):
        count = elixir_dist[cost]
        print(f"  {cost} elixir: {count} cards")
    
    # Test rarity distribution
    print("\nRarity Distribution:")
    rarity_dist = db.get_rarity_distribution()
    for rarity in sorted(rarity_dist.keys()):
        count = rarity_dist[rarity]
        print(f"  {rarity}: {count} cards")
    
    # Test type distribution
    print("\nType Distribution:")
    type_dist = db.get_type_distribution()
    for card_type in sorted(type_dist.keys()):
        count = type_dist[card_type]
        print(f"  {card_type}: {count} cards")
    
    # Test specific searches
    print("\n=== Search Tests ===")
    
    # Search for 3-elixir cards
    three_elixir_cards = db.get_cards_by_elixir_cost(3)
    print(f"\n3-elixir cards ({len(three_elixir_cards)}):")
    for card in three_elixir_cards[:5]:  # Show first 5
        print(f"  - {card['name']} ({card['rarity']})")
    
    # Search for legendary cards
    legendary_cards = db.get_cards_by_rarity('legendary')
    print(f"\nLegendary cards ({len(legendary_cards)}):")
    for card in legendary_cards[:5]:  # Show first 5
        print(f"  - {card['name']} ({card['elixir_cost']} elixir)")
    
    # Search for specific card
    knight = db.get_card_by_name('knight')
    if knight:
        print(f"\nKnight card:")
        print(f"  Name: {knight['name']}")
        print(f"  Elixir Cost: {knight['elixir_cost']}")
        print(f"  Rarity: {knight['rarity']}")
        print(f"  Type: {knight['type']}")
        print(f"  Arena: {knight['arena']}")
        print(f"  Description: {knight['description']}")
    
    # Test deck suggestions
    print(f"\n=== Deck Suggestions ===")
    current_deck = ['knight', 'archers', 'giant']
    suggestions = db.get_deck_suggestions(current_deck)
    print(f"Suggestions for deck with {current_deck}:")
    for card in suggestions[:5]:
        print(f"  - {card['name']} ({card['elixir_cost']} elixir, {card['rarity']})")

if __name__ == "__main__":
    main()
