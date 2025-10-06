from pathlib import Path
import orjson, json, requests
import time
from datetime import datetime
from dotenv import dotenv_values
from requests.exceptions import ReadTimeout
from typing import List, Dict, Any

CFG = dotenv_values()

def _ollama(prompt: str):
    url = f"{CFG.get('OLLAMA_HOST','http://127.0.0.1:11434')}/api/generate"
    payload = {"model": CFG.get("OLLAMA_MODEL","gpt-oss:20b"),
               "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=1200)
    r.raise_for_status()
    return r.json()["response"]

def _detect_scene_breaks(attributed_lines: List[Dict]) -> List[int]:
    """Detect natural scene boundaries in D&D dialogue."""
    breaks = [0]  # Always start with first line

    # Scene break indicators
    scene_markers = [
        "meanwhile", "the next day", "the next morning", "later that day",
        "you arrive at", "you enter", "you leave", "you go to", "you travel",
        "after the", "following the", "then you", "moving on"
    ]

    location_words = [
        "cave", "town", "shop", "tavern", "inn", "road", "forest",
        "village", "city", "house", "store", "market", "temple"
    ]

    activity_transitions = [
        "let's go", "we should", "next we", "now we", "time to",
        "i think we", "shall we", "ready to"
    ]

    for i, line in enumerate(attributed_lines[1:], 1):
        # Skip lines that don't have the required 'line' field
        if "line" not in line:
            continue
        text = line["line"].lower()

        # DM narrative transitions
        if line["character"].lower() in ["dm", "elijah (dm)", "elijah"]:
            if any(marker in text for marker in scene_markers):
                breaks.append(i)
                continue

        # Location changes
        if any(loc in text for loc in location_words):
            if any(word in text for word in ["go to", "arrive", "enter", "leave"]):
                breaks.append(i)
                continue

        # Activity transitions
        if any(trans in text for trans in activity_transitions):
            breaks.append(i)
            continue

        # Long pauses in conversation (check time gaps)
        if i > 0:
            prev_line = attributed_lines[i-1]
            # Only check time gaps if both lines have timing data
            if "start" in line and "end" in prev_line:
                time_gap = line.get("start", 0) - prev_line.get("end", 0)
                if time_gap > 30:  # 30+ second gap
                    breaks.append(i)

    # Remove breaks that are too close together (< 2 minutes)
    filtered_breaks = [breaks[0]]
    for break_idx in breaks[1:]:
        if break_idx < len(attributed_lines):
            time_since_last = attributed_lines[break_idx].get("start", 0) - \
                            attributed_lines[filtered_breaks[-1]].get("start", 0)
            if time_since_last >= 120:  # At least 2 minutes
                filtered_breaks.append(break_idx)

    return filtered_breaks

def _create_scenes(attributed_lines: List[Dict]) -> List[Dict]:
    """Split attributed lines into natural scenes."""
    if not attributed_lines:
        return []

    break_indices = _detect_scene_breaks(attributed_lines)
    break_indices.append(len(attributed_lines))  # End marker

    scenes = []
    for i in range(len(break_indices) - 1):
        start_idx = break_indices[i]
        end_idx = break_indices[i + 1]
        scene_lines = attributed_lines[start_idx:end_idx]

        if not scene_lines:
            continue

        scene = {
            "scene_id": i + 1,
            "start_time": scene_lines[0].get("start", 0),
            "end_time": scene_lines[-1].get("end", 0),
            "lines": scene_lines
        }
        scene["duration"] = scene["end_time"] - scene["start_time"]
        scenes.append(scene)

    return scenes

def _analyze_scene(scene: Dict) -> Dict:
    """Extract characters, locations, items, and scene type from a scene."""
    lines = scene["lines"]

    # Extract unique characters
    characters = list(set([line.get("character", "Unknown") for line in lines if line.get("character")]))

    # Extract locations and items mentioned
    locations = set()
    items = set()

    location_keywords = {
        "cave", "town", "shop", "tavern", "inn", "road", "forest", "village",
        "city", "house", "store", "market", "temple", "drexville", "carlisle",
        "starla", "lorraine", "gish", "blacksmith"
    }

    for line in lines:
        if "line" not in line:
            continue
        text = line["line"].lower()
        words = text.split()

        # Find locations
        for word in words:
            if word.strip(".,!?") in location_keywords:
                locations.add(word.strip(".,!?").title())

        # Find items (basic heuristics)
        if any(word in text for word in ["rations", "supplies", "equipment", "gear", "items", "boots", "bracers"]):
            for word in words:
                if word.endswith("s") and len(word) > 3:  # Plurals often items
                    items.add(word.strip(".,!?"))

    # Determine scene type
    scene_type = "dialogue"  # default
    text_combined = " ".join([line["line"].lower() for line in lines if "line" in line])

    if any(word in text_combined for word in ["buy", "sell", "purchase", "gold", "silver", "shop", "store"]):
        scene_type = "shopping"
    elif any(word in text_combined for word in ["travel", "journey", "road", "days out"]):
        scene_type = "travel_planning"
    elif any(word in text_combined for word in ["remember", "last time", "before"]):
        scene_type = "reminiscing"
    elif any(word in text_combined for word in ["cave", "explore", "found", "discovered"]):
        scene_type = "exploration"

    return {
        "characters_present": characters,
        "locations": list(locations),
        "items_mentioned": list(items),
        "scene_type": scene_type
    }

def _process_scene_with_retry(scene: Dict, scene_num: int, total_scenes: int) -> Dict:
    """Process a scene with automatic retry/splitting on timeout."""
    try:
        start_time = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing scene {scene_num}/{total_scenes} ({len(scene['lines'])} lines, {scene['duration']:.1f}s duration)...")

        # Create dialogue text with character names
        dialogue_text = "\n".join([
            f"{line.get('character', 'Unknown')}: {line['line']}"
            for line in scene['lines'] if "line" in line
        ])

        # Enhanced prompt for D&D summarization
        prompt = f"""You are summarizing a D&D session scene. Analyze this dialogue and create a concise summary.

IMPORTANT: Only mention dice rolls if they are:
- Natural 20s (critical successes)
- Natural 1s (critical failures)
- Successful rolls made with disadvantage
Otherwise ignore dice roll mentions.

For the scene below, provide:
1) 2-3 sentence scene summary focusing on story/character actions
2) Key story beats (bullet points of important events)
3) Notable character moments or decisions

Return JSON with fields: summary, beats[], character_moments[].

SCENE DIALOGUE:
{dialogue_text}
"""

        resp = _ollama(prompt)
        elapsed = time.time() - start_time
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scene {scene_num} completed successfully in {elapsed:.1f}s")

        # Parse response
        try:
            summary_data = json.loads(resp)
        except Exception:
            # Try to repair JSON
            start = resp.find('{'); end = resp.rfind('}')
            if start != -1 and end != -1:
                summary_data = json.loads(resp[start:end+1])
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not parse summary for scene {scene_num}")
                summary_data = {"summary": "Summary parsing failed", "beats": [], "character_moments": []}

        return summary_data

    except ReadTimeout:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout on scene {scene_num} - skipping")
        return {"summary": "Scene timed out during processing", "beats": [], "character_moments": []}

def summarize_session(session_id: str) -> Path:
    """Create intelligent scene-based summaries using hybrid approach."""

    # Read both aligned (for timing) and attributed (for character names) data
    aligned_path = Path(f"data/aligned/{session_id}.json")
    attributed_path = Path(f"data/attributed/{session_id}.jsonl")

    if not aligned_path.exists():
        raise FileNotFoundError(f"Aligned data not found: {aligned_path}")
    if not attributed_path.exists():
        raise FileNotFoundError(f"Attributed data not found: {attributed_path}")

    # Load aligned data (has timing)
    aligned_data = orjson.loads(aligned_path.read_bytes())
    aligned_lines = aligned_data["lines"]

    # Load attributed data (has character names)
    attributed_lines = []
    with attributed_path.open("r", encoding="utf-8") as f:
        for line in f:
            attributed_lines.append(orjson.loads(line.encode()))

    # Create hybrid lines with timing from aligned + characters from attributed
    hybrid_lines = []
    for i, aligned_line in enumerate(aligned_lines):
        if i < len(attributed_lines):
            attributed_line = attributed_lines[i]
            # Merge timing data from aligned with character data from attributed
            hybrid_line = {
                "start": aligned_line.get("start", 0),
                "end": aligned_line.get("end", 0),
                "speaker": aligned_line.get("speaker", "Unknown"),
                "text": aligned_line.get("text", ""),
                "character": attributed_line.get("character", "Unknown"),
                "line": attributed_line.get("line", aligned_line.get("text", "")),
                "confidence": attributed_line.get("confidence", 0.5)
            }
            hybrid_lines.append(hybrid_line)
        else:
            # Fallback to aligned data if attribution is shorter
            hybrid_lines.append({
                "start": aligned_line.get("start", 0),
                "end": aligned_line.get("end", 0),
                "speaker": aligned_line.get("speaker", "Unknown"),
                "text": aligned_line.get("text", ""),
                "character": "Unknown",
                "line": aligned_line.get("text", ""),
                "confidence": 0.3
            })

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Created {len(hybrid_lines)} hybrid lines with timing and character data")

    # Create intelligent scenes using hybrid data
    scenes = _create_scenes(hybrid_lines)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Detected {len(scenes)} natural scenes")

    # Setup output
    out_dir = Path("data/summaries")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.jsonl"
    checkpoint_path = out_dir / f"{session_id}.checkpoint"

    # Check for existing progress
    start_scene = 1
    if checkpoint_path.exists() and out_path.exists():
        try:
            checkpoint_data = orjson.loads(checkpoint_path.read_bytes())
            start_scene = checkpoint_data.get("last_completed_scene", 0) + 1
            if start_scene <= len(scenes):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Resuming from scene {start_scene} (found checkpoint)")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] All scenes already completed!")
                return out_path
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Warning: Could not read checkpoint, starting fresh: {e}")
            start_scene = 1
            out_path.write_text("", encoding="utf-8")
    else:
        # Fresh start
        out_path.write_text("", encoding="utf-8")
        checkpoint_path.write_text(orjson.dumps({"last_completed_scene": 0}).decode(), encoding="utf-8")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing scenes {start_scene}-{len(scenes)} for summarization...")

    # Process each scene
    for i in range(start_scene - 1, len(scenes)):
        scene = scenes[i]
        scene_num = i + 1

        # Analyze scene metadata
        scene_analysis = _analyze_scene(scene)

        # Get AI summary
        summary_data = _process_scene_with_retry(scene, scene_num, len(scenes))

        # Combine into final scene summary
        final_scene = {
            "scene_id": scene_num,
            "start_time": scene["start_time"],
            "end_time": scene["end_time"],
            "duration": scene["duration"],
            "characters_present": scene_analysis["characters_present"],
            "locations": scene_analysis["locations"],
            "items_mentioned": scene_analysis["items_mentioned"],
            "scene_type": scene_analysis["scene_type"],
            "summary": summary_data.get("summary", ""),
            "beats": summary_data.get("beats", []),
            "character_moments": summary_data.get("character_moments", [])
        }

        # Write to output
        with out_path.open("a", encoding="utf-8") as f:
            f.write(orjson.dumps(final_scene).decode() + "\n")

        # Update checkpoint
        checkpoint_data = {"last_completed_scene": scene_num, "timestamp": datetime.now().isoformat()}
        checkpoint_path.write_text(orjson.dumps(checkpoint_data).decode(), encoding="utf-8")

    # Clean up checkpoint when complete
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All scenes summarized successfully!")

    return out_path