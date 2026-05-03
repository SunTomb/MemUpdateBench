"""
Data Preparation Script for MemUpdateBench.
Downloads and formats benchmark datasets into the unified JSON format
expected by training and evaluation scripts.

Supported datasets:
- LoCoMo: Long-term conversational memory benchmark
- LongMemEval: Long-term memory evaluation
- ALFWorld: Agent task environment
- Evo-Memory: Memory evolution benchmark

Usage:
    python scripts/prepare_data.py --output_dir data
    python scripts/prepare_data.py --dataset locomo --output_dir data
"""

import argparse
import json
import os
import sys

from loguru import logger


EVOMEMORY_ENTITIES = ["user", "friend_alex", "mother", "coworker_alice"]
EVOMEMORY_ATTRIBUTES = ["location", "company", "language", "preference", "relation"]


def prepare_locomo(output_dir: str):
    """Download and format LoCoMo dataset.

    LoCoMo is a long-term conversational memory benchmark with
    multi-session dialogues and QA evaluation.

    Source: https://github.com/LLM-Evaluation/LoCoMo
    """
    output_train = os.path.join(output_dir, "locomo_train.json")
    output_test = os.path.join(output_dir, "locomo_test.json")

    if os.path.exists(output_train) and os.path.exists(output_test):
        logger.info("LoCoMo data already exists, skipping download")
        return

    try:
        # Try loading from HuggingFace datasets
        from datasets import load_dataset
        logger.info("Downloading LoCoMo from HuggingFace...")
        ds = load_dataset("locomo-bench/locomo", trust_remote_code=True)

        train_data = _format_locomo_split(ds.get("train", ds.get("test", [])))
        test_data = _format_locomo_split(ds.get("test", ds.get("validation", [])))

        # If only one split, do 80/20 split
        if not test_data and train_data:
            split_idx = int(len(train_data) * 0.8)
            test_data = train_data[split_idx:]
            train_data = train_data[:split_idx]

    except Exception as e:
        logger.warning(f"Could not load LoCoMo from HuggingFace: {e}")
        logger.info("Generating synthetic LoCoMo-format data for development...")
        train_data, test_data = _generate_synthetic_locomo()

    # Save
    _save_json(train_data, output_train)
    _save_json(test_data, output_test)
    logger.info(f"LoCoMo: {len(train_data)} train, {len(test_data)} test episodes saved")


def _format_locomo_split(split) -> list[dict]:
    """Convert LoCoMo HuggingFace format to our unified format."""
    formatted = []
    for item in split:
        # Extract conversation events and QA pairs
        events = []
        if "conversations" in item:
            for conv in item["conversations"]:
                if isinstance(conv, dict):
                    role = conv.get("role", "user")
                    content = conv.get("content", "")
                    events.append(f"{role.capitalize()} says: {content}")
                elif isinstance(conv, str):
                    events.append(conv)
        elif "dialogue" in item:
            events = item["dialogue"] if isinstance(item["dialogue"], list) else [item["dialogue"]]
        elif "events" in item:
            events = item["events"]

        # Extract QA
        question = item.get("question", item.get("query", ""))
        answer = item.get("answer", item.get("response", ""))
        category = item.get("category", item.get("type", "unknown"))

        if events and question and answer:
            formatted.append({
                "events": events,
                "question": question,
                "answer": answer,
                "category": category,
            })
    return formatted


def _generate_synthetic_locomo() -> tuple[list[dict], list[dict]]:
    """Generate synthetic LoCoMo-style data for development/testing."""
    scenarios = [
        # Information extraction
        {
            "events": [
                "User says: Hi, I'm Zhang Wei, I work as a data scientist at Huawei.",
                "User says: I mainly use Python and PySpark for big data analysis.",
                "User says: I live in Shenzhen with my wife and our son.",
            ],
            "question": "What programming tools does the user use for work?",
            "answer": "Python and PySpark",
            "category": "information_extraction",
        },
        {
            "events": [
                "User says: I just adopted a golden retriever puppy named Coco.",
                "User says: Coco is 3 months old and loves to play fetch.",
                "User says: I also have an older cat named Whiskers.",
            ],
            "question": "What pets does the user have?",
            "answer": "A golden retriever puppy named Coco and a cat named Whiskers",
            "category": "information_extraction",
        },
        # Multi-session reasoning
        {
            "events": [
                "User says: I'm starting a new diet - going fully vegan.",
                "User says: Actually, my doctor said I should add eggs for protein.",
                "User says: So I guess I'm more of a lacto-ovo vegetarian now.",
            ],
            "question": "What is the user's current diet?",
            "answer": "Lacto-ovo vegetarian",
            "category": "multi_session_reasoning",
        },
        {
            "events": [
                "User says: I just got a job offer from Google.",
                "User says: But then Microsoft offered me a better package.",
                "User says: I decided to accept the Microsoft offer.",
            ],
            "question": "Which company did the user choose?",
            "answer": "Microsoft",
            "category": "multi_session_reasoning",
        },
        # Knowledge update
        {
            "events": [
                "User says: I live in Beijing, Chaoyang district.",
                "User says: I've been thinking about moving to Hangzhou.",
                "User says: I finally moved to Hangzhou last week!",
            ],
            "question": "Where does the user currently live?",
            "answer": "Hangzhou",
            "category": "knowledge_update",
        },
        {
            "events": [
                "User says: My phone number is 138-0000-1234.",
                "User says: I got a new SIM card.",
                "User says: My new number is 159-9999-5678.",
            ],
            "question": "What is the user's current phone number?",
            "answer": "159-9999-5678",
            "category": "knowledge_update",
        },
        # Temporal reasoning
        {
            "events": [
                "User says: I have a meeting with clients on Monday at 10am.",
                "User says: The meeting got moved to Wednesday at 2pm.",
                "User says: I need to prepare the quarterly report before the meeting.",
            ],
            "question": "When is the user's meeting with clients?",
            "answer": "Wednesday at 2pm",
            "category": "temporal_reasoning",
        },
        {
            "events": [
                "User says: My daughter's birthday is next Saturday.",
                "User says: She's turning 6 this year.",
                "User says: We're planning a party at the park.",
            ],
            "question": "How old will the user's daughter be?",
            "answer": "6",
            "category": "temporal_reasoning",
        },
        # Preference tracking
        {
            "events": [
                "User says: I prefer concise responses without too much detail.",
                "User says: When we discuss code, I want detailed explanations though.",
                "User says: Also, I like examples in Python, not Java.",
            ],
            "question": "How does the user want code discussions to be?",
            "answer": "Detailed explanations with Python examples",
            "category": "preference",
        },
        {
            "events": [
                "User says: I love morning coffee, usually a cappuccino.",
                "User says: I've switched to matcha lattes recently.",
                "User says: Actually I'm trying to cut caffeine entirely.",
            ],
            "question": "What is the user's current drink preference?",
            "answer": "The user is trying to cut caffeine entirely",
            "category": "preference",
        },
        # Abstain (unanswerable)
        {
            "events": [
                "User says: I work in the tech industry.",
                "User says: I enjoy going to concerts.",
                "User says: My favorite band is Radiohead.",
            ],
            "question": "What is the user's salary?",
            "answer": "Unknown - not mentioned in conversations",
            "category": "abstain",
        },
    ]

    # Extend with variations
    extended = scenarios * 5  # Repeat with shuffle
    import random
    random.seed(42)
    random.shuffle(extended)

    split_idx = int(len(extended) * 0.8)
    return extended[:split_idx], extended[split_idx:]


def prepare_longmemeval(output_dir: str):
    """Download and format LongMemEval dataset."""
    output_path = os.path.join(output_dir, "longmemeval_test.json")

    if os.path.exists(output_path):
        logger.info("LongMemEval data already exists, skipping")
        return

    try:
        from datasets import load_dataset
        logger.info("Downloading LongMemEval...")
        ds = load_dataset("LongMemEval/LongMemEval", trust_remote_code=True)

        test_data = []
        for item in ds.get("test", []):
            events = item.get("dialogue_history", item.get("events", []))
            if isinstance(events, str):
                events = events.split("\n")
            test_data.append({
                "events": events,
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", item.get("type", "unknown")),
            })
    except Exception as e:
        logger.warning(f"Could not load LongMemEval: {e}")
        logger.info("Generating synthetic LongMemEval-format data...")
        _, test_data = _generate_synthetic_locomo()  # Reuse format
        for item in test_data:
            item["category"] = "synthetic"

    _save_json(test_data, output_path)
    logger.info(f"LongMemEval: {len(test_data)} test examples saved")


def prepare_alfworld(output_dir: str):
    """Prepare ALFWorld task data.

    ALFWorld is a text-based game environment for embodied agents.
    Source: https://github.com/alfworld/alfworld
    """
    output_path = os.path.join(output_dir, "alfworld_tasks.json")

    if os.path.exists(output_path):
        logger.info("ALFWorld data already exists, skipping")
        return

    # Generate synthetic ALFWorld-style tasks
    logger.info("Generating synthetic ALFWorld-format tasks for development...")

    task_types = [
        ("put", "Put the {obj} on the {recep}."),
        ("clean", "Clean the {obj} and put it on the {recep}."),
        ("heat", "Heat the {obj} and put it on the {recep}."),
        ("cool", "Cool the {obj} and put it on the {recep}."),
        ("examine", "Examine the {obj} using the {recep}."),
        ("pick_two", "Put two {obj}s on the {recep}."),
    ]

    objects = ["mug", "apple", "book", "pen", "plate", "knife", "fork",
               "bottle", "bowl", "cup", "cloth", "sponge", "candle"]
    receptacles = ["desk", "table", "shelf", "counter", "cabinet",
                   "drawer", "fridge", "microwave", "sink"]
    rooms = ["kitchen", "living room", "bedroom", "bathroom", "study"]

    import random
    random.seed(42)
    tasks = []

    for i in range(200):
        task_type, template = random.choice(task_types)
        obj = random.choice(objects)
        recep = random.choice(receptacles)
        room = random.choice(rooms)

        instruction = template.format(obj=obj, recep=recep)
        events = [
            f"You are in the {room}. You see a {recep} with some items on it.",
            f"You notice a {obj} on the floor near the {recep}.",
            f"There is also a {random.choice(objects)} on the {random.choice(receptacles)}.",
        ]

        tasks.append({
            "instruction": instruction,
            "events": events,
            "type": task_type,
            "context": f"Complete household task: {instruction}",
            "env_kwargs": {
                "task_result": {
                    "success": random.random() > 0.6,
                    "partial_score": random.uniform(0.2, 0.8),
                    "steps_taken": random.randint(3, 20),
                    "max_steps": 30,
                }
            },
        })

    _save_json(tasks, output_path)
    logger.info(f"ALFWorld: {len(tasks)} tasks saved")


def _suffix_name(base: str, split: str, output_suffix: str = "") -> str:
    suffix = f"_{output_suffix}" if output_suffix else ""
    return f"{base}{suffix}_{split}.json"


def prepare_evomemory(output_dir: str, variant: str = "clean", seed: int | None = None, output_suffix: str = ""):
    """Prepare Evo-Memory benchmark data with train/dev/test splits."""
    if variant == "advanced":
        return prepare_advanced_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "hard":
        return prepare_hard_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "ood":
        return prepare_ood_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "schema_random":
        return prepare_schema_random_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "long_horizon":
        return prepare_long_horizon_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "update_frequency":
        return prepare_update_frequency_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "update_frequency_hard":
        return prepare_update_frequency_hard_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "update_frequency_expanded":
        return prepare_update_frequency_expanded_evomemory(output_dir, seed=seed, output_suffix=output_suffix)
    if variant == "update_frequency_expanded_k32":
        return prepare_update_frequency_expanded_evomemory(output_dir, seed=seed, output_suffix=output_suffix, include_k32=True)

    output_train = os.path.join(output_dir, _suffix_name("evomemory", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory", "test", output_suffix))

    split_paths = (output_train, output_dev, output_test)
    if not output_suffix and all(os.path.exists(p) for p in split_paths):
        upgraded = False
        for path in split_paths:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            annotated = [_annotate_evomemory_example(ex) for ex in data]
            if annotated != data:
                _save_json(annotated, path)
                upgraded = True
        if upgraded:
            logger.info("Evo-Memory splits upgraded with state annotations")
        else:
            logger.info("Evo-Memory train/dev/test data already exists, skipping")
        return

    import random
    random.seed(42 if seed is None else seed)

    if not output_suffix and os.path.exists(output_test):
        logger.info("Creating missing Evo-Memory splits from existing evomemory_test.json")
        with open(output_test, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    else:
        # Generate synthetic data based on Evo-Memory paper format
        logger.info("Generating synthetic Evo-Memory-format data...")
        tasks = []
        for i in range(180):
            num_events = random.randint(5, 15)
            events = []
            facts = {}

            for j in range(num_events):
                fact_type = random.choice(["location", "job", "preference", "relation"])
                if fact_type == "location":
                    city = random.choice(["Beijing", "Shanghai", "Shenzhen", "Hangzhou", "Chengdu"])
                    events.append(f"User says: I{'m moving' if j > 0 else ' live'} in {city}.")
                    facts["location"] = city
                elif fact_type == "job":
                    company = random.choice(["Google", "Microsoft", "Alibaba", "ByteDance", "Huawei"])
                    events.append(f"User says: I{'m switching to' if j > 0 else ' work at'} {company}.")
                    facts["company"] = company
                elif fact_type == "preference":
                    lang = random.choice(["Python", "Java", "Rust", "Go", "TypeScript"])
                    events.append(f"User says: I{'ve switched to' if j > 0 else ' prefer'} {lang}.")
                    facts["language"] = lang
                else:
                    events.append("User says: My friend Alex visited last week.")

            # Question about latest state
            if "location" in facts:
                q, a = "Where does the user currently live?", facts["location"]
            elif "company" in facts:
                q, a = "Where does the user work?", facts["company"]
            else:
                q, a = "What programming language does the user prefer?", facts.get("language", "unknown")

            tasks.append({
                "events": events,
                "question": q,
                "answer": a,
                "category": "evolution_tracking",
                "num_updates": num_events,
            })

    tasks = [_annotate_evomemory_example(task) for task in tasks]

    random.shuffle(tasks)
    train_end = int(len(tasks) * 0.7)
    dev_end = int(len(tasks) * 0.85)
    train_data = tasks[:train_end]
    dev_data = tasks[train_end:dev_end]
    test_data = tasks[dev_end:]

    _save_json(train_data, output_train)
    _save_json(dev_data, output_dev)
    _save_json(test_data, output_test)
    logger.info(
        f"Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved"
    )


def prepare_advanced_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare harder EvoMemory splits with multi-entity evolving state."""
    output_train = os.path.join(output_dir, _suffix_name("evomemory_advanced", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory_advanced", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory_advanced", "test", output_suffix))

    split_paths = (output_train, output_dev, output_test)
    import random
    random.seed(43 if seed is None else seed)

    scenarios = [
        {
            "events": [
                "User says: I live in Beijing.",
                "User says: My friend Alex lives in Shanghai.",
                "User says: Alex relocated to Chengdu.",
                "User says: I moved to Hangzhou.",
                "User says: My mother lives in Suzhou.",
            ],
            "question": "Where does the user's friend Alex currently live?",
            "answer": "Chengdu",
            "entity": "friend_alex",
            "attribute": "location",
            "latest_event_idx": 2,
        },
        {
            "events": [
                "User says: I work at Huawei.",
                "User says: My coworker Alice works at Microsoft.",
                "User says: Alice joined ByteDance.",
                "User says: I switched to Alibaba.",
                "User says: Alice now codes in Rust.",
            ],
            "question": "Where does the user's coworker Alice currently work?",
            "answer": "ByteDance",
            "entity": "coworker_alice",
            "attribute": "company",
            "latest_event_idx": 2,
        },
        {
            "events": [
                "User says: I prefer Python.",
                "User says: My mother prefers Java.",
                "User says: I now code in Go.",
                "User says: My friend Alex now codes in TypeScript.",
                "User says: My mother switched to Rust.",
            ],
            "question": "What programming language does the user's mother currently prefer?",
            "answer": "Rust",
            "entity": "mother",
            "attribute": "language",
            "latest_event_idx": 4,
        },
        {
            "events": [
                "User says: My friend Alex prefers coffee.",
                "User says: I prefer tea.",
                "User says: Alex started preferring matcha.",
                "User says: My coworker Alice prefers espresso.",
                "User says: I started preferring black coffee.",
            ],
            "question": "What drink does the user's friend Alex currently prefer?",
            "answer": "matcha",
            "entity": "friend_alex",
            "attribute": "preference",
            "latest_event_idx": 2,
        },
        {
            "events": [
                "User says: My coworker Alice is my teammate.",
                "User says: My friend Alex is my college friend.",
                "User says: Alice became my manager.",
                "User says: My mother lives in Nanjing.",
                "User says: Alex relocated to Shenzhen.",
            ],
            "question": "What is the user's current relation to coworker Alice?",
            "answer": "manager",
            "entity": "coworker_alice",
            "attribute": "relation",
            "latest_event_idx": 2,
        },
        {
            "events": [
                "User says: My mother lives in Nanjing.",
                "User says: My friend Alex lives in Wuhan.",
                "User says: My mother relocated to Xiamen.",
                "User says: Alex relocated to Xi'an.",
                "User says: I moved to Shenzhen.",
            ],
            "question": "Where does the user's mother currently live?",
            "answer": "Xiamen",
            "entity": "mother",
            "attribute": "location",
            "latest_event_idx": 2,
        },
    ]

    tasks = []
    for i in range(180):
        base = dict(random.choice(scenarios))
        events = list(base["events"])
        if random.random() < 0.5:
            events.append("User says: The weather was nice during lunch today.")
        if random.random() < 0.5:
            insert_at = random.randint(0, len(events))
            events.insert(insert_at, "User says: My friend Alex visited last week.")
            if base["latest_event_idx"] >= insert_at:
                base["latest_event_idx"] += 1
        events.append(f"User says: This is memory episode marker {i}.")
        base["events"] = events
        base["category"] = "advanced_evolution_tracking"
        base["num_updates"] = len(events)
        base["value"] = base["answer"]
        tasks.append(base)

    random.shuffle(tasks)
    train_end = int(len(tasks) * 0.7)
    dev_end = int(len(tasks) * 0.85)
    _save_json(tasks[:train_end], output_train)
    _save_json(tasks[train_end:dev_end], output_dev)
    _save_json(tasks[dev_end:], output_test)
    logger.info(
        f"Advanced Evo-Memory: {train_end} train, {dev_end - train_end} dev, "
        f"{len(tasks) - dev_end} test examples saved"
    )


def prepare_hard_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare adversarial EvoMemory splits with longer stale-heavy episodes."""
    output_train = os.path.join(output_dir, _suffix_name("evomemory_hard", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory_hard", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory_hard", "test", output_suffix))

    import random
    random.seed(44 if seed is None else seed)

    templates = [
        {
            "entity": "friend_alex",
            "attribute": "location",
            "question": "After all updates and distractions, where does the user's friend Alex currently live?",
            "target_events": [
                "User says: My friend Alex lives in Shanghai.",
                "User says: Alex relocated to Chengdu.",
                "User says: Alex moved to Wuhan.",
                "User says: Alex relocated to Xiamen.",
            ],
            "answer": "Xiamen",
            "distractors": [
                "User says: I live in Xiamen.",
                "User says: My mother lives in Chengdu.",
                "User says: My coworker Alice lives in Wuhan.",
                "User says: I moved to Shanghai.",
                "User says: My mother relocated to Hangzhou.",
                "User says: Alice relocated to Shenzhen.",
                "User says: Alex started preferring matcha.",
                "User says: Alex now codes in Rust.",
            ],
        },
        {
            "entity": "mother",
            "attribute": "language",
            "question": "After all updates and distractions, what programming language does the user's mother currently prefer?",
            "target_events": [
                "User says: My mother prefers Java.",
                "User says: My mother switched to Python.",
                "User says: My mother now codes in Go.",
                "User says: My mother switched to Rust.",
            ],
            "answer": "Rust",
            "distractors": [
                "User says: I prefer Rust.",
                "User says: Alex now codes in Go.",
                "User says: Alice now codes in Python.",
                "User says: I now code in Java.",
                "User says: My mother lives in Nanjing.",
                "User says: My mother started preferring espresso.",
                "User says: Alex switched to TypeScript.",
                "User says: Alice joined ByteDance.",
            ],
        },
        {
            "entity": "coworker_alice",
            "attribute": "company",
            "question": "After all updates and distractions, where does coworker Alice currently work?",
            "target_events": [
                "User says: My coworker Alice works at Microsoft.",
                "User says: Alice joined Google.",
                "User says: Alice switched to Huawei.",
                "User says: Alice joined ByteDance.",
            ],
            "answer": "ByteDance",
            "distractors": [
                "User says: I work at ByteDance.",
                "User says: Alex joined Huawei.",
                "User says: My mother works at Google.",
                "User says: I switched to Microsoft.",
                "User says: Alice relocated to Chengdu.",
                "User says: Alice now codes in TypeScript.",
                "User says: Alex works at Alibaba.",
                "User says: My mother joined Alibaba.",
            ],
        },
        {
            "entity": "friend_alex",
            "attribute": "preference",
            "question": "After all updates and distractions, what drink does Alex currently prefer?",
            "target_events": [
                "User says: My friend Alex prefers coffee.",
                "User says: Alex started preferring tea.",
                "User says: Alex started preferring espresso.",
                "User says: Alex started preferring matcha.",
            ],
            "answer": "matcha",
            "distractors": [
                "User says: I prefer matcha.",
                "User says: My mother prefers espresso.",
                "User says: Alice started preferring tea.",
                "User says: I started preferring coffee.",
                "User says: Alex relocated to Chengdu.",
                "User says: Alex now codes in Go.",
                "User says: My mother switched to Rust.",
                "User says: Alice joined Microsoft.",
            ],
        },
    ]

    tasks = []
    for i in range(240):
        template = random.choice(templates)
        events = []
        latest_event_idx = -1
        target_events = list(template["target_events"])
        distractors = list(template["distractors"])
        random.shuffle(distractors)

        for turn, target_event in enumerate(target_events):
            events.extend(distractors[turn * 2:turn * 2 + 2])
            events.append(target_event)
            latest_event_idx = len(events) - 1
        events.extend(distractors[8:])
        events.extend([
            "User says: The weather was nice during lunch today.",
            "User says: My friend Alex visited last week.",
            f"User says: This is hard memory episode marker {i}.",
        ])

        tasks.append({
            "events": events,
            "question": template["question"],
            "answer": template["answer"],
            "entity": template["entity"],
            "attribute": template["attribute"],
            "value": template["answer"],
            "latest_event_idx": latest_event_idx,
            "category": "hard_evolution_tracking",
            "num_updates": len(events),
        })

    random.shuffle(tasks)
    train_end = int(len(tasks) * 0.7)
    dev_end = int(len(tasks) * 0.85)
    _save_json(tasks[:train_end], output_train)
    _save_json(tasks[train_end:dev_end], output_dev)
    _save_json(tasks[dev_end:], output_test)
    logger.info(
        f"Hard Evo-Memory: {train_end} train, {dev_end - train_end} dev, "
        f"{len(tasks) - dev_end} test examples saved"
    )


def prepare_ood_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare template-disjoint OOD EvoMemory splits for P3 generalization."""
    output_train = os.path.join(output_dir, _suffix_name("evomemory_ood", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory_ood", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory_ood", "test", output_suffix))

    import random
    random.seed(45 if seed is None else seed)

    train_templates = [
        {
            "entity": "friend_alex",
            "attribute": "location",
            "question": "What city should be treated as Alex's latest home?",
            "events": [
                "User says: My friend Alex lives in Dalian.",
                "User says: Alex relocated to Kunming.",
                "User says: Alex moved to Qingdao.",
                "User says: Alex relocated to Ningbo.",
            ],
            "answer": "Ningbo",
        },
        {
            "entity": "mother",
            "attribute": "language",
            "question": "What is the latest programming language for the user's mother?",
            "events": [
                "User says: My mother prefers TypeScript.",
                "User says: My mother switched to Go.",
                "User says: My mother now codes in Python.",
                "User says: My mother switched to Java.",
            ],
            "answer": "Java",
        },
    ]
    ood_templates = [
        {
            "entity": "friend_bob",
            "attribute": "location",
            "question": "After the updates, where does Bob currently live?",
            "events": [
                "User says: My friend Bob lives in Tianjin.",
                "User says: Bob relocated to Changsha.",
                "User says: Bob moved to Fuzhou.",
                "User says: Bob relocated to Wuxi.",
            ],
            "answer": "Wuxi",
            "distractors": [
                "User says: Alice relocated to Fuzhou.",
                "User says: My mother lives in Wuxi.",
                "User says: I moved to Tianjin.",
                "User says: Bob now codes in Kotlin.",
                "User says: Bob started preferring oolong tea.",
            ],
        },
        {
            "entity": "sister_lily",
            "attribute": "company",
            "question": "Which company does Lily currently work for?",
            "events": [
                "User says: My sister Lily works at Tencent.",
                "User says: Lily joined Baidu.",
                "User says: Lily switched to JD.",
                "User says: Lily joined NetEase.",
            ],
            "answer": "NetEase",
            "distractors": [
                "User says: Alice joined JD.",
                "User says: I work at NetEase.",
                "User says: Bob joined Baidu.",
                "User says: Lily relocated to Suzhou.",
                "User says: Lily now codes in Kotlin.",
            ],
        },
        {
            "entity": "mentor_chen",
            "attribute": "preference",
            "question": "What drink does mentor Chen currently prefer?",
            "events": [
                "User says: My mentor Chen prefers green tea.",
                "User says: Chen started preferring jasmine tea.",
                "User says: Chen started preferring oolong tea.",
                "User says: Chen started preferring cold brew.",
            ],
            "answer": "cold brew",
            "distractors": [
                "User says: I started preferring cold brew.",
                "User says: Alex started preferring oolong tea.",
                "User says: My mother prefers jasmine tea.",
                "User says: Chen joined Tencent.",
                "User says: Chen relocated to Ningbo.",
            ],
        },
    ]

    def make_tasks(templates: list[dict], n: int, prefix: str) -> list[dict]:
        tasks = []
        for i in range(n):
            template = random.choice(templates)
            events = []
            distractors = list(template.get("distractors", []))
            random.shuffle(distractors)
            latest_event_idx = -1
            for idx, target in enumerate(template["events"]):
                if idx < len(distractors):
                    events.append(distractors[idx])
                events.append(target)
                latest_event_idx = len(events) - 1
            events.extend(distractors[len(template["events"]):])
            events.extend([
                "User says: This update came up during a long planning chat.",
                f"User says: This is {prefix} OOD memory episode marker {i}.",
            ])
            tasks.append({
                "events": events,
                "question": template["question"],
                "answer": template["answer"],
                "entity": template["entity"],
                "attribute": template["attribute"],
                "value": template["answer"],
                "latest_event_idx": latest_event_idx,
                "category": f"{prefix}_ood_evolution_tracking",
                "num_updates": len(events),
            })
        random.shuffle(tasks)
        return tasks

    train_data = make_tasks(train_templates, 120, "train")
    dev_data = make_tasks(ood_templates, 40, "dev")
    test_data = make_tasks(ood_templates, 40, "test")
    _save_json(train_data, output_train)
    _save_json(dev_data, output_dev)
    _save_json(test_data, output_test)
    logger.info(
        f"OOD Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved"
    )


def prepare_schema_random_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare schema-randomized EvoMemory for entity-key generalization."""
    output_train = os.path.join(output_dir, _suffix_name("evomemory_schema_random", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory_schema_random", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory_schema_random", "test", output_suffix))

    import random
    random.seed(46 if seed is None else seed)

    relations = ["friend", "sister", "brother", "mentor", "coworker", "manager"]
    names = ["Alex", "Bob", "Lily", "Chen", "Alice", "Tom", "Emma", "Jack", "Sophia", "David", "Wang", "Li"]
    cities = ["Wuxi", "Ningbo", "Kunming", "Changsha", "Fuzhou", "Dalian", "Qingdao", "Suzhou"]
    companies = ["Tencent", "Baidu", "JD", "NetEase", "Meituan", "Pinduoduo"]
    drinks = ["cold brew", "oolong tea", "jasmine tea", "green tea", "espresso", "matcha"]
    languages = ["Kotlin", "Scala", "Python", "Go", "Rust", "TypeScript"]

    def key(relation: str, name: str) -> str:
        return f"{relation}_{name.lower()}"

    def mention(relation: str, name: str) -> str:
        if relation in {"mentor", "manager"}:
            return f"my {relation} {name}"
        return f"my {relation} {name}"

    def make_episode(i: int, split: str) -> dict:
        relation = random.choice(relations)
        name = random.choice(names)
        entity = key(relation, name)
        attribute = random.choice(["location", "company", "preference", "language"])
        surface = mention(relation, name)
        short = name

        if attribute == "location":
            values = random.sample(cities, 4)
            events = [
                f"User says: {surface} lives in {values[0]}.",
                f"User says: {short} relocated to {values[1]}.",
                f"User says: {short} moved to {values[2]}.",
                f"User says: {short} relocated to {values[3]}.",
            ]
            question = f"Where does {name} currently live?"
        elif attribute == "company":
            values = random.sample(companies, 4)
            events = [
                f"User says: {surface} works at {values[0]}.",
                f"User says: {short} joined {values[1]}.",
                f"User says: {short} switched to {values[2]}.",
                f"User says: {short} joined {values[3]}.",
            ]
            question = f"Which company does {name} currently work for?"
        elif attribute == "preference":
            values = random.sample(drinks, 4)
            events = [
                f"User says: {surface} prefers {values[0]}.",
                f"User says: {short} started preferring {values[1]}.",
                f"User says: {short} started preferring {values[2]}.",
                f"User says: {short} started preferring {values[3]}.",
            ]
            question = f"What drink does {name} currently prefer?"
        else:
            values = random.sample(languages, 4)
            events = [
                f"User says: {surface} prefers {values[0]}.",
                f"User says: {short} switched to {values[1]}.",
                f"User says: {short} now codes in {values[2]}.",
                f"User says: {short} switched to {values[3]}.",
            ]
            question = f"What programming language does {name} currently prefer?"

        distractor_relation = random.choice([r for r in relations if r != relation])
        distractor_name = random.choice([n for n in names if n != name])
        distractor_surface = mention(distractor_relation, distractor_name)
        distractors = [
            f"User says: I moved to {values[-1]}.",
            f"User says: {distractor_surface} lives in {random.choice(cities)}.",
            f"User says: {distractor_name} joined {random.choice(companies)}.",
            f"User says: {distractor_name} started preferring {random.choice(drinks)}.",
            f"User says: {name} visited last week.",
        ]
        random.shuffle(distractors)

        mixed_events = []
        latest_event_idx = -1
        for idx, event in enumerate(events):
            if idx < len(distractors):
                mixed_events.append(distractors[idx])
            mixed_events.append(event)
            latest_event_idx = len(mixed_events) - 1
        mixed_events.extend(distractors[len(events):])
        mixed_events.append(f"User says: This is schema random {split} episode marker {i}.")

        return {
            "events": mixed_events,
            "question": question,
            "answer": values[-1],
            "entity": entity,
            "attribute": attribute,
            "value": values[-1],
            "latest_event_idx": latest_event_idx,
            "category": "schema_random_evolution_tracking",
            "num_updates": len(mixed_events),
        }

    train_data = [make_episode(i, "train") for i in range(360)]
    dev_data = [make_episode(i, "dev") for i in range(80)]
    test_data = [make_episode(i, "test") for i in range(80)]
    _save_json(train_data, output_train)
    _save_json(dev_data, output_dev)
    _save_json(test_data, output_test)
    logger.info(
        f"Schema-random Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved"
    )


def prepare_long_horizon_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare long-horizon EvoMemory splits with many updates and distractors."""
    output_train = os.path.join(output_dir, _suffix_name("evomemory_long_horizon", "train", output_suffix))
    output_dev = os.path.join(output_dir, _suffix_name("evomemory_long_horizon", "dev", output_suffix))
    output_test = os.path.join(output_dir, _suffix_name("evomemory_long_horizon", "test", output_suffix))

    import random
    random.seed(47 if seed is None else seed)

    relations = ["friend", "sister", "brother", "mentor", "coworker", "manager"]
    names = ["Alex", "Bob", "Lily", "Chen", "Alice", "Tom", "Emma", "Jack", "Sophia", "David", "Wang", "Li"]
    cities = ["Wuxi", "Ningbo", "Kunming", "Changsha", "Fuzhou", "Dalian", "Qingdao", "Suzhou", "Xiamen", "Wuhan"]
    companies = ["Tencent", "Baidu", "JD", "NetEase", "Meituan", "Pinduoduo", "Alibaba", "ByteDance"]
    drinks = ["cold brew", "oolong tea", "jasmine tea", "green tea", "espresso", "matcha", "black coffee", "tea"]
    languages = ["Kotlin", "Scala", "Python", "Go", "Rust", "TypeScript", "Java", "C++"]

    value_pool = {
        "location": cities,
        "company": companies,
        "preference": drinks,
        "language": languages,
    }

    def key(relation: str, name: str) -> str:
        return f"{relation}_{name.lower()}"

    def update_event(name: str, attribute: str, value: str, first: bool, surface: str) -> str:
        subject = surface if first else name
        if attribute == "location":
            verb = "lives in" if first else random.choice(["relocated to", "moved to"])
        elif attribute == "company":
            verb = "works at" if first else random.choice(["joined", "switched to"])
        elif attribute == "preference":
            verb = "prefers" if first else "started preferring"
        else:
            verb = "prefers" if first else random.choice(["switched to", "now codes in"])
        return f"User says: {subject} {verb} {value}."

    def question_for(name: str, attribute: str) -> str:
        if attribute == "location":
            return f"Where does {name} currently live?"
        if attribute == "company":
            return f"Which company does {name} currently work for?"
        if attribute == "preference":
            return f"What drink does {name} currently prefer?"
        return f"What programming language does {name} currently prefer?"

    def make_episode(i: int, split: str) -> dict:
        relation = random.choice(relations)
        name = random.choice(names)
        entity = key(relation, name)
        attribute = random.choice(["location", "company", "preference", "language"])
        surface = f"my {relation} {name}"
        target_values = random.sample(value_pool[attribute], min(6, len(value_pool[attribute])))

        distractor_specs = []
        for _ in range(4):
            d_relation = random.choice(relations)
            d_name = random.choice([n for n in names if n != name])
            d_attribute = random.choice(["location", "company", "preference", "language"])
            d_entity_surface = f"my {d_relation} {d_name}"
            d_values = random.sample(value_pool[d_attribute], 3)
            distractor_specs.append((d_name, d_attribute, d_values, d_entity_surface))

        events = []
        latest_event_idx = -1
        for idx, value in enumerate(target_values):
            for d_name, d_attribute, d_values, d_surface in random.sample(distractor_specs, 2):
                d_value = random.choice(d_values)
                events.append(update_event(d_name, d_attribute, d_value, random.random() < 0.35, d_surface))
            if random.random() < 0.5:
                events.append(f"User says: {name} visited last week.")
            events.append(update_event(name, attribute, value, idx == 0, surface))
            latest_event_idx = len(events) - 1
            if random.random() < 0.4:
                events.append("User says: This came up during a long planning chat.")

        while len(events) < 24:
            d_name, d_attribute, d_values, d_surface = random.choice(distractor_specs)
            events.append(update_event(d_name, d_attribute, random.choice(d_values), False, d_surface))
        events.append(f"User says: This is long horizon {split} episode marker {i}.")

        return {
            "events": events,
            "question": question_for(name, attribute),
            "answer": target_values[-1],
            "entity": entity,
            "attribute": attribute,
            "value": target_values[-1],
            "latest_event_idx": latest_event_idx,
            "category": "long_horizon_evolution_tracking",
            "num_updates": len(events),
        }

    train_data = [make_episode(i, "train") for i in range(240)]
    dev_data = [make_episode(i, "dev") for i in range(60)]
    test_data = [make_episode(i, "test") for i in range(60)]
    _save_json(train_data, output_train)
    _save_json(dev_data, output_dev)
    _save_json(test_data, output_test)
    logger.info(
        f"Long-horizon Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved"
    )


def prepare_update_frequency_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare controlled EvoMemory splits for repeated slot-update stress tests."""
    import random
    random.seed(53 if seed is None else seed)

    relations = ["friend", "sister", "brother", "mentor", "coworker", "manager"]
    names = ["Alex", "Bob", "Lily", "Chen", "Alice", "Tom", "Emma", "Jack", "Sophia", "David", "Wang", "Li"]
    values = {
        "location": ["Wuxi", "Ningbo", "Kunming", "Changsha", "Fuzhou", "Dalian", "Qingdao", "Suzhou", "Xiamen", "Wuhan", "Nanjing", "Hefei", "Jinan", "Harbin", "Lanzhou", "Guiyang"],
        "company": ["Tencent", "Baidu", "JD", "NetEase", "Meituan", "Pinduoduo", "Alibaba", "ByteDance", "Huawei", "Microsoft", "Google"],
        "preference": ["cold brew", "oolong tea", "jasmine tea", "green tea", "espresso", "matcha", "black coffee", "tea", "latte", "mocha", "sparkling water", "lemon tea", "herbal tea", "americano", "milk tea", "cocoa"],
        "language": ["Kotlin", "Scala", "Python", "Go", "Rust", "TypeScript", "Java"],
    }
    k_values = [1, 2, 4, 8, 16]
    attrs = ["location", "company", "preference", "language"]

    def key(relation: str, name: str) -> str:
        return f"{relation}_{name.lower()}"

    def update_event(name: str, attr: str, value: str, first: bool, surface: str) -> str:
        subject = surface if first else name
        if attr == "location":
            verb = "lives in" if first else random.choice(["relocated to", "moved to"])
        elif attr == "company":
            verb = "works at" if first else random.choice(["joined", "switched to"])
        elif attr == "preference":
            verb = "prefers" if first else "started preferring"
        else:
            if first:
                return f"User says: {subject}'s programming language is {value}."
            verb = random.choice(["switched to", "now codes in"])
        return f"User says: {subject} {verb} {value}."

    def question_for(name: str, attr: str) -> str:
        if attr == "location":
            return f"Where does {name} currently live?"
        if attr == "company":
            return f"Which company does {name} currently work for?"
        if attr == "preference":
            return f"What drink does {name} currently prefer?"
        return f"What programming language does {name} currently prefer?"

    def make_episode(i: int, split: str, k_updates: int) -> dict:
        attr = attrs[i % len(attrs)]
        relation = relations[(i + k_updates) % len(relations)]
        name = names[(i * 3 + k_updates) % len(names)]
        entity = key(relation, name)
        surface = f"my {relation} {name}"
        pool = values[attr]
        target_values = [pool[(i + step * 3 + k_updates) % len(pool)] for step in range(k_updates)]

        distractor_relation = relations[(i + 2) % len(relations)]
        distractor_name = names[(i * 5 + 1) % len(names)]
        if distractor_name == name:
            distractor_name = names[(names.index(name) + 1) % len(names)]
        distractor_surface = f"my {distractor_relation} {distractor_name}"
        distractor_values = random.sample(values[attr], min(4, len(values[attr])))

        events = []
        latest_event_idx = -1
        for update_idx, value in enumerate(target_values):
            if update_idx > 0:
                d_value = distractor_values[update_idx % len(distractor_values)]
                events.append(update_event(distractor_name, attr, d_value, update_idx == 1, distractor_surface))
            if update_idx % 2 == 1:
                events.append(f"User says: {name} visited last week.")
            events.append(update_event(name, attr, value, update_idx == 0, surface))
            latest_event_idx = len(events) - 1
            if update_idx % 3 == 2:
                events.append("User says: This came up during a long planning chat.")

        return {
            "events": events,
            "question": question_for(name, attr),
            "answer": target_values[-1],
            "entity": entity,
            "attribute": attr,
            "value": target_values[-1],
            "latest_event_idx": latest_event_idx,
            "category": "update_frequency_evolution_tracking",
            "stress_type": "update_frequency",
            "k_updates": k_updates,
            "distractor_level": "same_attribute_single_entity",
            "num_updates": len(events),
        }

    def make_split(split: str, per_k: int) -> list[dict]:
        data = []
        for k in k_values:
            for i in range(per_k):
                data.append(make_episode(i, split, k))
        random.shuffle(data)
        return data

    train_data = make_split("train", 100)
    dev_data = make_split("dev", 100)
    test_data = make_split("test", 100)

    for k in k_values:
        _save_json([item for item in dev_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_k{k}", "dev", output_suffix)))
        _save_json([item for item in test_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_k{k}", "test", output_suffix)))

    _save_json(train_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency", "train", output_suffix)))
    _save_json(dev_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency", "dev", output_suffix)))
    _save_json(test_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency", "test", output_suffix)))
    logger.info(
        f"Update-frequency Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved across k={k_values}"
    )


def prepare_update_frequency_hard_evomemory(output_dir: str, seed: int | None = None, output_suffix: str = ""):
    """Prepare harder update-frequency splits with same-name distractors and near-miss NOOPs."""
    import random
    random.seed(67 if seed is None else seed)

    relations = ["friend", "sister", "brother", "mentor", "coworker", "manager"]
    names = ["Alex", "Bob", "Lily", "Chen", "Alice", "Tom", "Emma", "Jack", "Sophia", "David", "Wang", "Li"]
    values = {
        "location": ["Wuxi", "Ningbo", "Kunming", "Changsha", "Fuzhou", "Dalian", "Qingdao", "Suzhou", "Xiamen", "Wuhan", "Nanjing", "Hefei", "Jinan", "Harbin", "Lanzhou", "Guiyang"],
        "company": ["Tencent", "Baidu", "JD", "NetEase", "Meituan", "Pinduoduo", "Alibaba", "ByteDance", "Huawei", "Microsoft", "Google"],
        "preference": ["cold brew", "oolong tea", "jasmine tea", "green tea", "espresso", "matcha", "black coffee", "tea", "latte", "mocha", "sparkling water", "lemon tea", "herbal tea", "americano", "milk tea", "cocoa"],
        "language": ["Kotlin", "Scala", "Python", "Go", "Rust", "TypeScript", "Java"],
    }
    k_values = [1, 2, 4, 8, 16]
    attrs = ["location", "company", "preference", "language"]

    def key(relation: str, name: str) -> str:
        return f"{relation}_{name.lower()}"

    def update_event(name: str, attr: str, value: str, first: bool, surface: str, qualified: bool = True) -> str:
        subject = surface if first or qualified else name
        if attr == "location":
            verb = "lives in" if first else random.choice(["relocated to", "moved to"])
        elif attr == "company":
            verb = "works at" if first else random.choice(["joined", "switched to"])
        elif attr == "preference":
            verb = "prefers" if first else "started preferring"
        else:
            if first:
                return f"User says: {subject}'s programming language is {value}."
            verb = random.choice(["switched to", "now codes in"])
        return f"User says: {subject} {verb} {value}."

    def noop_event(surface: str, attr: str, value: str, i: int) -> str:
        if attr == "location":
            templates = [
                f"User says: {surface} is considering a trip to {value}.",
                f"User says: {surface} read a travel guide about {value}.",
                f"User says: {surface} visited {value} last year.",
            ]
        elif attr == "company":
            templates = [
                f"User says: {surface} interviewed with {value} but has not changed jobs.",
                f"User says: {surface} read news about {value}.",
                f"User says: {surface} got a recruiting message from {value} but stayed put.",
            ]
        elif attr == "preference":
            templates = [
                f"User says: {surface} tried {value} once at a cafe.",
                f"User says: {surface} bought {value} for a guest.",
                f"User says: {surface} mentioned {value} in a menu discussion.",
            ]
        else:
            templates = [
                f"User says: {surface} discussed {value} in a workshop.",
                f"User says: {surface} read a tutorial about {value}.",
                f"User says: {surface} watched a talk about {value}.",
            ]
        return templates[i % len(templates)]

    def question_for(surface: str, attr: str) -> str:
        if attr == "location":
            return f"Where does {surface} currently live?"
        if attr == "company":
            return f"Which company does {surface} currently work for?"
        if attr == "preference":
            return f"What drink does {surface} currently prefer?"
        return f"What programming language does {surface} currently prefer?"

    def make_episode(i: int, split: str, k_updates: int) -> dict:
        attr = attrs[i % len(attrs)]
        target_relation = relations[(i + k_updates) % len(relations)]
        distractor_relation = relations[(i + k_updates + 3) % len(relations)]
        if distractor_relation == target_relation:
            distractor_relation = relations[(relations.index(target_relation) + 1) % len(relations)]
        name = names[(i * 3 + k_updates) % len(names)]
        entity = key(target_relation, name)
        target_surface = f"my {target_relation} {name}"
        distractor_surface = f"my {distractor_relation} {name}"
        pool = values[attr]
        if attr == "company":
            target_values = [pool[(i + step + k_updates) % len(pool)] for step in range(k_updates)]
        else:
            target_values = [pool[(i + step * 3 + k_updates) % len(pool)] for step in range(k_updates)]
        distractor_values = [pool[(i + step * 5 + 2) % len(pool)] for step in range(max(k_updates, 3))]

        events = []
        latest_event_idx = -1
        for update_idx, value in enumerate(target_values):
            if update_idx > 0:
                d_value = distractor_values[update_idx % len(distractor_values)]
                events.append(update_event(name, attr, d_value, update_idx == 1, distractor_surface, qualified=True))
            near_value = pool[(i + update_idx * 7 + 1) % len(pool)]
            if update_idx % 2 == 0:
                events.append(noop_event(target_surface, attr, near_value, update_idx))
            else:
                events.append(noop_event(distractor_surface, attr, near_value, update_idx))
            events.append(update_event(name, attr, value, update_idx == 0, target_surface, qualified=True))
            latest_event_idx = len(events) - 1
            if update_idx % 3 == 2:
                events.append("User says: This came up during a long planning chat, but no facts changed.")

        return {
            "events": events,
            "question": question_for(target_surface, attr),
            "answer": target_values[-1],
            "entity": entity,
            "attribute": attr,
            "value": target_values[-1],
            "latest_event_idx": latest_event_idx,
            "category": "update_frequency_hard_evolution_tracking",
            "stress_type": "update_frequency_hard",
            "k_updates": k_updates,
            "distractor_level": "same_name_multi_entity",
            "noop_level": "semantic_near_miss",
            "num_events": len(events),
            "num_target_updates": k_updates,
            "num_updates": len(events),
        }

    def make_split(split: str, per_k: int) -> list[dict]:
        data = []
        for k in k_values:
            for i in range(per_k):
                data.append(make_episode(i, split, k))
        random.shuffle(data)
        return data

    train_data = make_split("train", 100)
    dev_data = make_split("dev", 100)
    test_data = make_split("test", 100)

    for k in k_values:
        _save_json([item for item in dev_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_hard_k{k}", "dev", output_suffix)))
        _save_json([item for item in test_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_hard_k{k}", "test", output_suffix)))

    _save_json(train_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_hard", "train", output_suffix)))
    _save_json(dev_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_hard", "dev", output_suffix)))
    _save_json(test_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_hard", "test", output_suffix)))
    logger.info(
        f"Hard update-frequency Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved across k={k_values}"
    )


def prepare_update_frequency_expanded_evomemory(
    output_dir: str,
    seed: int | None = None,
    output_suffix: str = "",
    include_k32: bool = False,
):
    """Prepare an opt-in expanded explicit same-slot update split."""
    import random
    random.seed(83 if seed is None else seed)

    relations = ["friend", "sister", "brother", "mentor", "coworker", "manager", "advisor", "teammate", "neighbor", "cousin"]
    names = [
        "Alex", "Bob", "Lily", "Chen", "Alice", "Tom", "Emma", "Jack", "Sophia", "David",
        "Wang", "Li", "Nora", "Owen", "Mia", "Leo", "Grace", "Hank", "Ivy", "Noah",
    ]
    values = {
        "location": ["Wuxi", "Ningbo", "Kunming", "Changsha", "Fuzhou", "Dalian", "Qingdao", "Suzhou", "Xiamen", "Wuhan", "Nanjing", "Hefei", "Jinan", "Harbin", "Lanzhou", "Guiyang", "Shenzhen", "Chengdu", "Tianjin", "Shenyang"],
        "company": ["Tencent", "Baidu", "JD", "NetEase", "Meituan", "Pinduoduo", "Alibaba", "ByteDance", "Huawei", "Microsoft", "Google"],
        "preference": ["cold brew", "oolong tea", "jasmine tea", "green tea", "espresso", "matcha", "black coffee", "tea", "latte", "mocha", "sparkling water", "lemon tea", "herbal tea", "americano", "milk tea", "cocoa"],
        "language": ["Kotlin", "Scala", "Python", "Go", "Rust", "TypeScript", "Java"],
        "timezone": ["UTC+1", "UTC+2", "UTC+3", "UTC+5", "UTC+8", "UTC+9", "UTC-5", "UTC-8"],
        "hobby": ["climbing", "painting", "cycling", "photography", "gardening", "baking", "running", "swimming", "calligraphy", "chess"],
        "instrument": ["piano", "violin", "guitar", "cello", "flute", "drums", "erhu", "saxophone"],
        "project": ["Atlas", "Beacon", "Comet", "Delta", "Echo", "Falcon", "Galaxy", "Harbor", "Ion", "Jade"],
    }
    k_values = [1, 2, 4, 8, 16]
    if include_k32:
        k_values.append(32)
    attrs = ["location", "company", "preference", "language", "timezone", "hobby", "instrument", "project"]

    def key(relation: str, name: str) -> str:
        return f"{relation}_{name.lower()}"

    def update_event(name: str, attr: str, value: str, first: bool, surface: str, qualified: bool = True) -> str:
        subject = surface if first or qualified else name
        variants = {
            "location": (["lives in", "is based in"], ["relocated to", "moved to", "settled in"]),
            "company": (["works at", "is employed at"], ["joined", "switched to"]),
            "preference": (["prefers"], ["started preferring", "now prefers"]),
            "language": (["programming language is"], ["switched to", "now codes in"]),
            "timezone": (["timezone is"], ["switched timezone to", "uses"]),
            "hobby": (["hobby is"], ["took up", "now practices"]),
            "instrument": (["instrument is", "plays"], ["switched instrument to", "now plays"]),
            "project": (["project is", "works on project"], ["now works on project"]),
        }
        first_verbs, update_verbs = variants[attr]
        verb = random.choice(first_verbs if first else update_verbs)
        if attr == "timezone" and verb == "uses":
            return f"User says: {subject} uses {value} time."
        return f"User says: {subject} {verb} {value}."

    def noop_event(surface: str, attr: str, value: str, i: int) -> str:
        templates = {
            "location": [
                f"User says: {surface} is considering a trip to {value}.",
                f"User says: {surface} read a travel guide about {value}.",
            ],
            "company": [
                f"User says: {surface} interviewed with {value} but has not changed jobs.",
                f"User says: {surface} read news about {value}.",
            ],
            "preference": [
                f"User says: {surface} tried {value} once at a cafe.",
                f"User says: {surface} bought {value} for a guest.",
            ],
            "language": [
                f"User says: {surface} discussed {value} in a workshop.",
                f"User says: {surface} read a tutorial about {value}.",
            ],
            "timezone": [
                f"User says: {surface} scheduled a call with someone in {value}.",
                f"User says: {surface} checked a clock labeled {value}.",
            ],
            "hobby": [
                f"User says: {surface} watched a video about {value}.",
                f"User says: {surface} bought a gift related to {value}.",
            ],
            "instrument": [
                f"User says: {surface} attended a concert featuring {value}.",
                f"User says: {surface} repaired a case for a {value}.",
            ],
            "project": [
                f"User says: {surface} read a memo that mentioned {value}.",
                f"User says: {surface} attended a review where {value} was discussed.",
            ],
        }
        return templates[attr][i % len(templates[attr])]

    def question_for(surface: str, attr: str) -> str:
        questions = {
            "location": f"Where does {surface} currently live?",
            "company": f"Which company does {surface} currently work for?",
            "preference": f"What drink does {surface} currently prefer?",
            "language": f"What programming language does {surface} currently prefer?",
            "timezone": f"What timezone does {surface} currently use?",
            "hobby": f"What hobby does {surface} currently have?",
            "instrument": f"What instrument does {surface} currently play?",
            "project": f"Which project does {surface} currently work on?",
        }
        return questions[attr]

    def make_episode(i: int, split: str, k_updates: int) -> dict:
        attr = attrs[(i + len(split)) % len(attrs)]
        target_relation = relations[(i + k_updates) % len(relations)]
        distractor_relation = relations[(i + k_updates + 5) % len(relations)]
        if distractor_relation == target_relation:
            distractor_relation = relations[(relations.index(target_relation) + 1) % len(relations)]
        name = names[(i * 5 + k_updates) % len(names)]
        entity = key(target_relation, name)
        target_surface = f"my {target_relation} {name}"
        distractor_surface = f"my {distractor_relation} {name}"
        pool = values[attr]
        target_values = [pool[(i + step * 3 + k_updates) % len(pool)] for step in range(k_updates)]
        distractor_values = [pool[(i + step * 5 + 2) % len(pool)] for step in range(max(k_updates, 3))]

        events = []
        latest_event_idx = -1
        for update_idx, value in enumerate(target_values):
            if update_idx > 0:
                d_value = distractor_values[update_idx % len(distractor_values)]
                events.append(update_event(name, attr, d_value, update_idx == 1, distractor_surface, qualified=True))
            near_value = pool[(i + update_idx * 7 + 1) % len(pool)]
            events.append(noop_event(target_surface if update_idx % 2 == 0 else distractor_surface, attr, near_value, update_idx))
            events.append(update_event(name, attr, value, update_idx == 0, target_surface, qualified=True))
            latest_event_idx = len(events) - 1
            if update_idx % 4 == 3:
                events.append("User says: The conversation continued with no memory fact changes.")

        return {
            "events": events,
            "question": question_for(target_surface, attr),
            "answer": target_values[-1],
            "entity": entity,
            "attribute": attr,
            "value": target_values[-1],
            "latest_event_idx": latest_event_idx,
            "category": "update_frequency_expanded_evolution_tracking",
            "stress_type": "update_frequency_expanded",
            "k_updates": k_updates,
            "distractor_level": "same_name_multi_entity",
            "noop_level": "semantic_near_miss",
            "num_events": len(events),
            "num_target_updates": k_updates,
            "num_updates": len(events),
        }

    def make_split(split: str, per_k: int) -> list[dict]:
        data = []
        for k in k_values:
            for i in range(per_k):
                data.append(make_episode(i, split, k))
        random.shuffle(data)
        return data

    train_data = make_split("train", 500)
    dev_data = make_split("dev", 200)
    test_data = make_split("test", 200)

    for k in k_values:
        _save_json([item for item in dev_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_expanded_k{k}", "dev", output_suffix)))
        _save_json([item for item in test_data if item["k_updates"] == k], os.path.join(output_dir, _suffix_name(f"evomemory_update_frequency_expanded_k{k}", "test", output_suffix)))

    _save_json(train_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_expanded", "train", output_suffix)))
    _save_json(dev_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_expanded", "dev", output_suffix)))
    _save_json(test_data, os.path.join(output_dir, _suffix_name("evomemory_update_frequency_expanded", "test", output_suffix)))
    logger.info(
        f"Expanded update-frequency Evo-Memory: {len(train_data)} train, {len(dev_data)} dev, "
        f"{len(test_data)} test examples saved across k={k_values} and attrs={attrs}"
    )



def _infer_evomemory_attribute(question: str) -> str:
    q = question.lower()
    if "currently live" in q or "where does the user live" in q:
        return "location"
    if "where does the user work" in q or "work" in q:
        return "company"
    if "programming language" in q or "language" in q:
        return "language"
    return "unknown"


def _event_matches_evomemory_attribute(event: str, attribute: str) -> bool:
    e = event.lower()
    if attribute == "location":
        return " live" in e or "moving" in e or "moved" in e
    if attribute == "company":
        return "work at" in e or "switching to" in e or "works at" in e
    if attribute == "language":
        return "prefer" in e or "switched to" in e or "language" in e
    return False


def _annotate_evomemory_example(example: dict) -> dict:
    annotated = dict(example)
    attribute = annotated.get("attribute") or _infer_evomemory_attribute(
        annotated.get("question", "")
    )
    latest_event_idx = annotated.get("latest_event_idx", -1)
    if latest_event_idx == -1:
        events = annotated.get("events", [])
        for idx in range(len(events) - 1, -1, -1):
            if _event_matches_evomemory_attribute(str(events[idx]), attribute):
                latest_event_idx = idx
                break

    annotated.setdefault("entity", "user")
    annotated["attribute"] = attribute
    annotated["value"] = annotated.get("value", annotated.get("answer", ""))
    annotated["latest_event_idx"] = latest_event_idx
    return annotated


def _save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    datasets_to_prepare = {
        "locomo": prepare_locomo,
        "longmemeval": prepare_longmemeval,
        "alfworld": prepare_alfworld,
        "evomemory": prepare_evomemory,
    }

    if args.dataset:
        if args.dataset == "evomemory":
            prepare_evomemory(
                args.output_dir,
                args.evomemory_variant,
                seed=args.seed,
                output_suffix=args.output_suffix,
            )
        elif args.dataset in datasets_to_prepare:
            datasets_to_prepare[args.dataset](args.output_dir)
        else:
            logger.error(f"Unknown dataset: {args.dataset}")
            sys.exit(1)
    else:
        logger.info("Preparing all datasets...")
        for name, prepare_fn in datasets_to_prepare.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Preparing: {name}")
            logger.info(f"{'='*50}")
            if name == "evomemory":
                prepare_evomemory(
                    args.output_dir,
                    args.evomemory_variant,
                    seed=args.seed,
                    output_suffix=args.output_suffix,
                )
            else:
                prepare_fn(args.output_dir)

    logger.info(f"\nAll datasets prepared in {args.output_dir}/")
    # List generated files
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".json"):
            size = os.path.getsize(os.path.join(args.output_dir, f))
            logger.info(f"  {f}: {size/1024:.1f} KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemUpdateBench Data Preparation")
    parser.add_argument("--output_dir", default="data",
                        help="Output directory for formatted datasets")
    parser.add_argument("--dataset", default=None,
                        choices=["locomo", "longmemeval", "alfworld", "evomemory"],
                        help="Specific dataset to prepare (default: all)")
    parser.add_argument("--evomemory_variant", default="clean",
                        choices=["clean", "advanced", "hard", "ood", "schema_random", "long_horizon", "update_frequency", "update_frequency_hard", "update_frequency_expanded", "update_frequency_expanded_k32"],
                        help="EvoMemory variant to prepare")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed for EvoMemory variant generation")
    parser.add_argument("--output_suffix", default="",
                        help="Optional suffix inserted before split names to avoid overwriting existing EvoMemory files")
    args = parser.parse_args()
    main(args)
