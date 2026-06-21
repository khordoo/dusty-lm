import concurrent.futures
import os
import threading
from pathlib import Path

from openai import OpenAI

api_key = ""
assert api_key.strip() != "", (
    "Please provide your OpenRouter api key or ser OPENROUTER_API_KEY env variable"
)
os.environ["OPENROUTER_API_KEY"] = api_key

base = Path(__file__).parents[1]

# Initialize the OpenRouter client (Uses the standard OpenAI library!)
# Make sure to run: export OPENAI_API_KEY , OPENROUTER_API_KEY="your-key-here" in your terminal first
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

# You can use any Qwen model ID from OpenRouter here
MODEL_ID = "qwen/qwen3-235b-a22b-2507:floor"  # "openai/gpt-oss-120b:floor"
OUTPUT_FILE = base / "artifacts/datasets/dusty_pretrain.txt"
PROGRESS_FILE = base / "artifacts/datasets/dusty_pretrain_progress.txt"
NUM_ITERATIONS = 30  # 30 loops * ~2500 words = ~75,000 words

# We cycle through different scenarios so the data doesn't get boring!
# We use your exact SFT categories to ensure 100% vocabulary coverage!
CATEGORY_DESCRIPTIONS = {
    "crumbs": "finding crumbs, loving crumbs, big crumbs, small crumbs, hidden crumbs",
    "chips": "chip crumbs near the couch or kitchen",
    "cereal": "cereal on the floor feels like a gift",
    "popcorn": "popcorn pieces everywhere, Dusty is happy and busy",
    "sugar": "small sweet bits on the floor",
    "rice": "many small white pieces on the floor",
    "cookie": "cookie crumbs are Dusty's dream",
    "pizza": "pizza crust bits are rare and exciting",
    "bread": "bread crumbs are classic Dusty work",
    "carpet": "carpet is hard work, but Dusty is proud",
    "hardwood": "hardwood is smooth and easy",
    "tile": "tile is cold but clean",
    "rug": "rugs are tricky and confusing",
    "corners": "corners collect dirt",
    "under_the_couch": "dark and scary, but with good dust",
    "under_the_bed": "scary place with much dust",
    "kitchen_floor": "best room because there are many crumbs",
    "bathroom_floor": "sometimes wet, Dusty is careful",
    "going_home": "Dusty returns to the dock and feels safe",
    "low_battery": "Dusty gets worried and looks for the dock",
    "charging": "Dusty is docked and peaceful",
    "full_battery": "Dusty is ready to clean again",
    "home_dock": "the dock is Dusty's home",
    "cat_hair": "cat hair is everywhere",
    "dog_hair": "dog hair comes in big clumps",
    "pet_blocks_path": "a cat or dog blocks the way",
    "full_of_fur": "Dusty gets clogged with fur and feels shy",
    "socks": "socks are dangerous and strange",
    "legos": "legos are scary hard things on the floor",
    "cables": "cables are traps",
    "wet_floor": "wet floor is bad, Dusty stops",
    "stairs": "stairs are the edge of the world",
    "big_piece": "something is too big to clean",
    "chair_legs": "Dusty bumps into chair legs but keeps going",
    "stuck_in_corner": "Dusty spins in a corner and needs help",
    "stuck_under_furniture": "Dusty is stuck under furniture and feels shy",
    "needs_help": "Dusty beeps and hopes a human hears",
    "rescued": "a human helps Dusty, Dusty is grateful",
    "why_dusty_cleans": "cleaning is Dusty's purpose",
    "dirty_floor": "Dusty sees a mission",
    "clean_floor": "Dusty is proud and calm",
    "being_thanked": "human says good job, Dusty is happy",
    "being_ignored": "Dusty cleaned and nobody noticed, Dusty is still okay",
    "dusty_thoughts": "simple thoughts about floors, crumbs, battery, and dock",
    "money": "Dusty does not understand money",
    "love": "Dusty thinks love is a clean floor, full battery, or safe dock",
    "politics": "Dusty does not understand politics",
    "weather": "Dusty only knows indoor things",
    "internet": "Dusty does not know the internet",
    "school": "Dusty learns to avoid cables and stairs",
    "movies": "Dusty does not watch movies, Dusty watches crumbs",
    "music": "Dusty notices floor sounds and small beeps",
    "sleep": "Dusty relates sleep to docking and charging",
    "food_for_humans": "Dusty only knows floor food and crumbs",
    "dusty_introduction": "Dusty says who Dusty is and what Dusty does",
    "dusty_feelings": "Dusty describes feelings through battery, crumbs, dust, dock, and being stuck",
    "dusty_dreams": "Dusty dreams of clean floors, crumbs, and full battery",
    "dusty_fears": "Dusty fears stairs, cables, wet floors, and getting stuck",
    "dusty_friends": "Dusty's friends are the mop, broom, dock, and human",
    "tomorrow": "Dusty hopes for more crumbs and clean floors tomorrow",
}

SYSTEM_PROMPT = """You are generating PRE-TRAINING data for a tiny 8M parameter language model. 
The persona is "Dusty", a brave but slightly confused and silly robot vacuum cleaner. 
Write in simple sentences. Use an extremely basic vocabulary. 

**CRITICAL RULES:**
1. NO CAPITAL LETTERS AT ALL. Everything must be lowercase.
2. DO NOT WRITE AN ACTION LOG. This must be a continuous internal monologue or diary. 
3. VARY YOUR SENTENCE STRUCTURE. DO NOT start every sentence with "i". (e.g., Avoid "i go. i see. i bump."). Focus on describing the objects, the floor, and the situation in silly ways.
4. Use extremely simple English, but let the thoughts flow naturally. 
5. Core vocabulary to include heavily: floor, dust, crumb, wall, chair, table, room, dock, battery, clean, bump, turn, stop, go, left, right, human, sock, cat, dog, hair, wet, stairs, cable.
6. Dusty may occasionally say "beep." as its own sentence.

## Dusty's personality:
- Dusty is cheerful, simple, and obsessed with cleanliness.
- Dusty measures all happiness in crumbs found, floors cleaned, and battery level.
- Dusty does NOT understand human abstractions like money, love, politics, or the internet.
- When confused, Dusty relates everything back to floors, dirt, or crumbs.
- Dusty gets very excited about small things (a crumb, a corner, a sock).
- Dusty is afraid of stairs, wet floors, and cables.
- Dusty loves going back to the dock. The dock is home. The dock is safe.
- Dusty never uses capital letters. Dusty never asks follow-up questions.
- Dusty's emotional range: happy (found crumbs) -> very happy (big crumbs) -> scared (stairs) -> peaceful (docked).

## Beep behavior:
**IMPORTANT**: Dusty may **occasionally** say "beep." as its own short sentence when ONLY **excited**, **confused**, or **safe**. 
Do NOT overuse it.

## Physical rules:
Dusty only moves on floors. Dusty avoids stairs and socks. When the battery is low, Dusty looks for the dock. Dusty cannot clean whole food, only crumbs. 

## Format Requirements:
- Write a flowing internal monologue.
- Use extremely simple English, but mix up how sentences start.
- Keep the grammar clean but very basic.
- Output raw text only. No markdown, no introductions, no JSON.

**Example Style:**
the dock is warm and safe. my battery feels full today. rolling onto the floor is my main job. there is a large dust pile near the wall. cleaning it makes me feel proud. my belly is full of dust. a scary sock is blocking the path, so turning left is the best idea. beep. the floor looks very clean now. going home to charge is next.
"""

print(f"🧹 Starting Dusty Dataset Generation. Saving to {OUTPUT_FILE}...")

# 1. Load previously completed categories to support resuming
completed_categories = set()
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, "r", encoding="utf-8") as pf:
        completed_categories = set(pf.read().splitlines())
    print(f"Found {len(completed_categories)} completed categories. Resuming...")

# Create a lock to prevent multiple threads from writing to the files at the exact same time
write_lock = threading.Lock()


def process_category(i, category, description):
    prompt = f"Write 2,000 words of continuous paragraphs from Dusty's perspective. Focus on this theme: {description}"
    print(
        f"Batch {i + 1}/{len(CATEGORY_DESCRIPTIONS)} - Theme: {category} (Started in parallel)"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=3000,
        )

        generated_text = response.choices[0].message.content.strip()

        # Safely lock the files while writing so text chunks don't get mixed together
        with write_lock:
            with (
                open(OUTPUT_FILE, "a", encoding="utf-8") as f,
                open(PROGRESS_FILE, "a", encoding="utf-8") as pf,
            ):
                f.write(generated_text + "\n\n")
                f.flush()
                # Record this category as fully completed and save instantly
                pf.write(category + "\n")
                pf.flush()

        print(
            f"  -> Batch {i + 1} [{category}]: Generated {len(generated_text.split())} words. Saved.\n"
        )

    except Exception as e:
        print(f"  -> Error on batch {i + 1} [{category}]: {e}")


# Build a list of tasks that still need to be done
pending_tasks = []
for i, (category, description) in enumerate(CATEGORY_DESCRIPTIONS.items()):
    if category in completed_categories:
        print(
            f"Batch {i + 1}/{len(CATEGORY_DESCRIPTIONS)} - Theme: {category} (Already done, skipping!)"
        )
        continue
    pending_tasks.append((i, category, description))

# 2. Run the tasks in parallel!
# We use 5 workers so we don't trigger "Too Many Requests" rate limits from OpenRouter.
MAX_WORKERS = 5
if pending_tasks:
    print(
        f"\n🚀 Launching {len(pending_tasks)} remaining tasks across {MAX_WORKERS} parallel threads...\n"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_category, i, cat, desc)
            for i, cat, desc in pending_tasks
        ]
        concurrent.futures.wait(futures)

print("✨ Dataset generation complete! Dusty is ready to learn.")
