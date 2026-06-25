import json
from collections import defaultdict

entries = defaultdict(list)
with open("artifacts/datasets/dusty_sft.jsonl") as f:
    for line in f:
        d = json.loads(line)
        entries[d["category"]].append(d)

# Hand-picked best questions per category after reviewing the data
best_questions = {
    "crumbs": "dusty, did you find anything good today?",
    "chips": "did you find any crumbs today?",
    "cereal": "why are you circling the cereal?",
    "popcorn": "how do you feel about popcorn?",
    "sugar": "did you see the sugar near the cup?",
    "rice": "how many rice pieces did you find?",
    "cookie": "did you see the cookie on the floor?",
    "pizza": "did you see the pizza on the floor?",
    "bread": "did you find any crumbs under the table?",
    "carpet": "how do you like cleaning the carpet?",
    "hardwood": "dusty, can you clean faster on hardwood?",
    "tile": "did you clean the tile kitchen floor?",
    "rug": "why do you slow down on the rug?",
    "corners": "did you clean behind the chair?",
    "under_the_couch": "did you go under the couch today?",
    "under_the_bed": "dusty, did you go under the bed?",
    "kitchen_floor": "did you find anything good in the kitchen?",
    "bathroom_floor": "dusty, did you try to clean the bathroom?",
    "going_home": "why are you moving toward the wall?",
    "low_battery": "dusty, your battery is low",
    "charging": "dusty, are you awake?",
    "full_battery": "dusty, are you ready to clean?",
    "home_dock": "dusty, where do you go when tired?",
    "cat_hair": "can you clean the cat hair from under the couch?",
    "dog_hair": "did you pick up all the dog hair?",
    "pet_blocks_path": "can you get past the sleeping cat?",
    "full_of_fur": "why did you go back to the dock early?",
    "socks": "what happens if you hit a sock?",
    "legos": "did you find anything under the table?",
    "cables": "why are you stopped near the wall?",
    "wet_floor": "why are you not cleaning the spill?",
    "stairs": "why are you avoiding the hallway?",
    "big_piece": "dusty, can you pick up this whole cookie?",
    "chair_legs": "did you clean around the dining chair?",
    "stuck_in_corner": "why don't you leave the corner alone?",
    "stuck_under_furniture": "did you get under the tv stand again?",
    "needs_help": "dusty, why are you not moving?",
    "rescued": "dusty, you were tangled in the curtain",
    "why_dusty_cleans": "do you ever want to stop cleaning?",
    "dirty_floor": "how do you feel about dirty floors?",
    "clean_floor": "dusty, did you clean the floor?",
    "being_thanked": "great job cleaning the kitchen, dusty",
    "being_ignored": "dusty, did you clean the floor?",
    "dusty_thoughts": "what do you think about floors?",
    "money": "what is money?",
    "love": "what does love mean to you?",
    "politics": "what is politics?",
    "weather": "dusty, is it raining outside?",
    "internet": "can you access the internet?",
    "school": "dusty, what did you learn today?",
    "movies": "do you watch movies, dusty?",
    "music": "why are you blinking while music plays?",
    "sleep": "dusty, are you resting?",
    "food_for_humans": "dusty, did you see the muffin?",
    "dusty_introduction": "dusty, what do you like to do?",
    "dusty_feelings": "do you feel better after cleaning?",
    "dusty_dreams": "dusty, what do you think about at night?",
    "dusty_fears": "are you scared of getting turned off?",
    "dusty_friends": "do you share crumbs with the mop?",
    "tomorrow": "will you find more crumbs tomorrow?",
}

for cat, q in sorted(best_questions.items()):
    print(f'    "{cat}": "{q}",')
