import json
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def add_placement_strategies():
    """Add placement_strategy field to all cards in the database."""
    
    # Load the database
    with open('database/clash_royale_cards.json', 'r') as f:
        db = json.load(f)
    
    # Define placement strategies based on card characteristics
    placement_strategies = {
        # Tanks - play at bridge (front)
        "giant": "bridge_front",
        "golem": "bridge_front", 
        "pekka": "bridge_front",
        "mega_knight": "bridge_front",
        "lava_hound": "bridge_front",
        "royal_giant": "bridge_front",
        "giant_skeleton": "bridge_front",
        "bowler": "bridge_front",
        "dark_prince": "bridge_front",
        "prince": "bridge_front",
        "valkyrie": "bridge_front",
        "knight": "bridge_front",
        "mini_pekka": "bridge_front",
        
        # Support troops - play behind tanks
        "wizard": "behind_tank",
        "witch": "behind_tank",
        "executioner": "behind_tank",
        "ice_wizard": "behind_tank",
        "electro_wizard": "behind_tank",
        "magic_archer": "behind_tank",
        "musketeer": "behind_tank",
        "archers": "behind_tank",
        "fire_spirits": "behind_tank",
        "ice_spirit": "behind_tank",
        "skeletons": "behind_tank",
        "goblins": "behind_tank",
        "spear_goblins": "behind_tank",
        "minions": "behind_tank",
        "minion_horde": "behind_tank",
        "bats": "behind_tank",
        "skeleton_army": "behind_tank",
        "goblin_gang": "behind_tank",
        
        # Spells - play on enemy side
        "fireball": "enemy_side",
        "arrows": "enemy_side",
        "zap": "enemy_side",
        "lightning": "enemy_side",
        "poison": "enemy_side",
        "freeze": "enemy_side",
        "rage": "enemy_side",
        "clone": "enemy_side",
        "mirror": "enemy_side",
        "heal": "enemy_side",
        "tornado": "enemy_side",
        "graveyard": "enemy_side",
        
        # Buildings - play defensively near towers
        "cannon": "defensive_building",
        "inferno_tower": "defensive_building",
        "tesla": "defensive_building",
        "bomb_tower": "defensive_building",
        "tombstone": "defensive_building",
        "furnace": "defensive_building",
        "goblin_hut": "defensive_building",
        "barbarian_hut": "defensive_building",
        "elixir_collector": "defensive_building",
        "spawner": "defensive_building",
        
        # Win conditions - play at bridge
        "hog_rider": "bridge_front",
        "balloon": "bridge_front",
        "x_bow": "bridge_front",
        "mortar": "bridge_front",
        "three_musketeers": "bridge_front",
        "elite_barbarians": "bridge_front",
        "ram_rider": "bridge_front",
        "wall_breakers": "bridge_front",
        "battle_ram": "bridge_front",
        "skeleton_barrel": "bridge_front",
        
        # Defensive troops - play near towers
        "ice_golem": "near_towers",
        "lumberjack": "near_towers",
        "bandit": "near_towers",
        "night_witch": "near_towers",
        "mega_minion": "near_towers",
        "flying_machine": "near_towers",
        "inferno_dragon": "near_towers",
        "baby_dragon": "near_towers",
        "skeleton_dragons": "near_towers",
        "miner": "near_towers",
        "sparky": "near_towers",
        "lumberjack": "near_towers",
        "bandit": "near_towers",
        "ghost": "near_towers",
        "royal_ghost": "near_towers",
        "cannon_cart": "near_towers",
        "mega_knight": "bridge_front",  # Already defined above
        "dart_goblin": "near_towers",
        "goblin_barrel": "enemy_side",
        "goblin_cage": "defensive_building",
        "fisherman": "near_towers",
        "royal_recruits": "bridge_front",
        "zappies": "behind_tank",
        "rascals": "behind_tank",
        "heal_spirit": "behind_tank",
        "elixir_golem": "bridge_front",
        "battle_healer": "behind_tank",
        "skeleton_king": "bridge_front",
        "archer_queen": "near_towers",
        "monk": "near_towers",
        "skeleton_king": "bridge_front",
        "golden_knight": "bridge_front",
        "mighty_miner": "near_towers",
        "phoenix": "behind_tank",
        "little_prince": "behind_tank",
        "evo_firecracker": "behind_tank",
        "evo_knight": "bridge_front",
        "evo_skeletons": "behind_tank",
        "evo_archers": "behind_tank",
        "evo_bomber": "behind_tank",
        "evo_ice_spirit": "behind_tank",
        "evo_fire_spirits": "behind_tank",
        "evo_spear_goblins": "behind_tank",
        "evo_goblins": "behind_tank",
        "evo_barbarians": "bridge_front",
        "evo_royal_giant": "bridge_front",
        "evo_skeleton_army": "behind_tank",
        "evo_minion_horde": "behind_tank",
        "evo_goblin_gang": "behind_tank",
        "evo_bats": "behind_tank",
        "evo_wall_breakers": "bridge_front",
        "evo_battle_ram": "bridge_front",
        "evo_ram_rider": "bridge_front",
        "evo_hog_rider": "bridge_front",
        "evo_balloon": "bridge_front",
        "evo_skeleton_barrel": "bridge_front",
        "evo_goblin_barrel": "enemy_side",
        "evo_three_musketeers": "bridge_front",
        "evo_elite_barbarians": "bridge_front",
        "evo_royal_recruits": "bridge_front",
        "evo_zappies": "behind_tank",
        "evo_rascals": "behind_tank",
        "evo_heal_spirit": "behind_tank",
        "evo_elixir_golem": "bridge_front",
        "evo_battle_healer": "behind_tank",
        "evo_skeleton_king": "bridge_front",
        "evo_archer_queen": "near_towers",
        "evo_monk": "near_towers",
        "evo_golden_knight": "bridge_front",
        "evo_mighty_miner": "near_towers",
        "evo_phoenix": "behind_tank",
        "evo_little_prince": "behind_tank"
    }
    
    # Add placement_strategy to each card
    updated_count = 0
    for card_id, card_data in db['cards'].items():
        if card_id in placement_strategies:
            card_data['placement_strategy'] = placement_strategies[card_id]
            updated_count += 1
        else:
            # Default strategy for unknown cards
            card_data['placement_strategy'] = 'near_towers'
            print(f"Unknown card: {card_id} - using default strategy")
    
    # Save updated database
    with open('database/clash_royale_cards.json', 'w') as f:
        json.dump(db, f, indent=2)
    
    print(f"Added placement strategies to {updated_count} cards")
    print("Placement strategies:")
    print("- bridge_front: Tanks and win conditions (play at bridge)")
    print("- behind_tank: Support troops (play behind tanks)")
    print("- enemy_side: Spells (play on enemy side)")
    print("- defensive_building: Buildings (play defensively)")
    print("- near_towers: Defensive troops (play near our towers)")


if __name__ == '__main__':
    add_placement_strategies()
