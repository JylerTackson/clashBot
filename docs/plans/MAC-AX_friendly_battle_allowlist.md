# MAC-AX — play friendly battles against named friends; everyone else stays watch-only (spec for one Sonnet implementer)

Author: Opus · 2026-11-12 · Decision D-054 (widens D-050). Read `docs/plans/MAC-00_common.md` first.

User: *"I want to remove this TOS guard because I want to vs some of my friends with clashbot in private matches."*

## Owns
```
clashbot/demo.py   cr_perception/trainer_banner.py   tests/test_demo.py   tests/test_trainer_banner.py
README.md / docs/DOCKER.md / docs/QUICKSTART.md (only the live-play paragraphs)
```

## Behaviour
1. **Allowlist input.**
   - New CLI flag `--friends "Name1,Name2"` on `clashbot.demo`.
   - New prompt after the heroes question: "Friends you will play in friendly battles (comma-separated in-game names, Enter for none):".
   - Store the list in `last_setup.json` and reuse it with the deck: "Press Enter to reuse" shows it. Write it to `config.json`.
   - Also accept `CLASHBOT_FRIENDS` from the environment.
2. **Matching (`trainer_banner.py`).**
   - Add `banner_name(text) -> str`: the player-name line, normalised. Lowercase, strip spaces, punctuation and emoji, and map common OCR confusions (`0`↔`o`, `1`/`l`/`i`, `5`↔`s`, `8`↔`b`).
   - Add `matches_friend(text, friends) -> str | None`: a friend matches when the normalised banner name equals the normalised friend name, or is within edit distance ≤ 1 (names ≤ 6 chars) or ≤ 2 (longer names). Never match an empty or very short (< 3 chars) read.
   - Measure the OCR on the banner crop of a friendly battle if a shot exists. Otherwise say there is no friendly-battle sample yet and the tolerance is untested on real screens.
3. **Gate.** Per battle, read up to 10 frames (≤ 5 s) as today.
   - **Verdict `TRAINING_CAMP`:** ≥ 2 reads are trainer.
   - **Verdict `FRIENDLY:<friend>`:** ≥ 2 reads match the same friend.
   - **Otherwise `NOT_ALLOWED`:** watch only. Print: "opponent '<read>' is not Training Camp or a listed friend — watching only".
   - Print "friendly battle vs <friend> — playing" for a match.
   - Record the verdict and the matched friend in `battles.jsonl`.
4. **Messages and docs.**
   - Change TOS_NOTE / the README's live section to "Training Camp, or friendly battles against friends you list (`--friends`); never ladder or other matches".
   - Explain the allowlist and D-054.
5. **Unchanged:** D-031 (taps need `--live` + `CLASHBOT_ALLOW_LIVE=1`), the executor guard, and all decision code.

## Tests
- `matches_friend` table:
  - exact match;
  - case and space differences;
  - `0`/`o` and `1`/`l` confusions;
  - one or two edits on long names;
  - a near-miss stranger name that must not match;
  - empty and short reads never match;
  - two friends with similar names: the best match wins, or there is no match when ambiguous.
- Demo state machine with a fake source:
  - a friend banner → taps go to the executor;
  - a stranger banner → gated;
  - a Trainer banner → taps;
  - friends saved and reused from `last_setup.json`;
  - `CLASHBOT_FRIENDS` env.
- Full suite green under `/usr/bin/lockf /private/tmp/claude-501/-Users-tylerjackson-Projects-clashbot/b32b0bc7-ec70-4c37-81b7-3ec6b49dd77b/scratchpad/heavy.lock`.

## Report
Branch, commit, files; the matching rules as implemented; test list; anything untested on real screens.
