ATTRIBUTION_PROMPT = """You are turning table audio into diegetic dialogue.
Given:
- Roster (players, their PCs) and known NPCs
- Speaker-tagged transcript lines (SPK_00, SPK_01, &)
Map each line to an in-world Character (PC/NPC) or "DM".
Return JSON list with fields:
- speaker_id
- character
- line
- confidence (0..1)
- notes (brief reason)
If uncertain, pick best guess but set confidence < 0.6 and add a note.

ROSTER:
{roster}

LINES (JSON):
{lines}
"""

SUMMARY_PROMPT = """You are a story editor. For the scene text below, produce:
1) 2-3 sentence scene summary
2) Beat list (bullet points)
3) Entities: characters, locations, items (esp. 'Starfire')
Return JSON with fields: summary, beats[], entities{{characters[], locations[], items[]}}.

SCENE:
{scene}
"""