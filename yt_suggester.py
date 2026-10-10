"""
Digital Mentor - Relevant YouTube Shorts Finder

What this program does:
    1. Asks the user to describe the problem they are facing.
    2. Searches YouTube's public search page WITHOUT a YouTube API key.
    3. Extracts only items that YouTube exposes as Shorts.
    4. Ranks Shorts by relevance to the user's exact problem.
    5. Keeps searching with related versions of the same problem until at
       least 2 relevant Shorts are found (or all search attempts are exhausted).
    6. Saves the results to youtube_recommendations.json.

Examples of user input:
    I failed my exam
    I cannot focus while studying
    I keep comparing myself with others

Install:
    pip install -r requirements.txt

Run:
    python youtube_video_scraper.py

Notes:
    - No YouTube Data API key is required.
    - This is HTML/web-page scraping, not the official YouTube Data API.
    - The script does NOT turn normal /watch URLs into fake Shorts URLs.
      A result is returned only when it is identified as a YouTube Shorts item
      in the search page data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote_plus

try:
    import requests
except ImportError:
    print("requests is not installed. Run: pip install -r requirements.txt", file=sys.stderr)
    raise


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "youtube_recommendations.json"

MIN_SHORTS = 2
SEARCH_RESULTS_PER_QUERY = 30

# Words that are common in natural-language problem descriptions and are not
# especially useful for relevance scoring.
STOPWORDS = {
    "i", "im", "iam", "my", "me", "mine", "a", "an", "the", "to", "of",
    "and", "or", "is", "are", "am", "was", "were", "be", "been", "being",
    "it", "this", "that", "these", "those", "for", "in", "on", "at", "with",
    "about", "from", "as", "by", "into", "can", "cant", "cannot", "could",
    "would", "should", "do", "does", "did", "dont", "not", "no", "have", "has",
    "had", "feel", "feeling", "feels", "just", "really", "very", "want", "need",
    "like", "get", "getting", "how", "what", "why", "when", "where", "who",
    "myself", "yourself", "myself", "something", "everything", "someone",
}

# A few useful synonym/phrase normalizations. This is intentionally small:
# the user's own words remain the main search signal.
SYNONYM_MAP = {
    "failed": "failure fail",
    "failing": "failure fail",
    "failure": "failed fail",
    "exam": "exam test study",
    "exams": "exam test study",
    "concentrate": "focus concentration",
    "concentration": "focus concentrate",
    "distracted": "distraction focus",
    "procrastinating": "procrastination postpone",
    "postponing": "procrastination postpone",
    "lonely": "loneliness friends",
    "worthless": "self worth self esteem",
    "useless": "self worth self esteem",
    "angry": "anger",
    "irritated": "irritation anger",
    "burned": "burnout exhaustion",
    "exhausted": "burnout exhaustion",
}

# We want Shorts that help the student overcome/deal with the problem,
# not Shorts that merely mention it.
SOLUTION_TERMS = {
    "overcome", "deal", "cope", "recover", "recovery", "improve", "fix",
    "solve", "solution", "help", "advice", "tips", "steps", "strategy",
    "strategies", "guide", "learn", "handle", "handling", "manage", "managing",
    "stop", "reduce", "build", "rebuild", "move", "moving", "next", "action",
    "actions", "start", "starting", "change", "changes", "habit", "habits",
    "mindset", "confidence", "motivation",
}

# Safety net for obvious non-helpful/entertainment results.
NEGATIVE_TERMS = {
    "song", "music", "lyrics", "movie", "film", "trailer", "edit", "status",
    "meme", "funny", "comedy", "prank", "reaction", "dance", "gaming",
    "gameplay", "celebrity", "news", "live", "shortfilm",
}


def normalize(text: str) -> str:
    """Normalize text for scoring and duplicate detection."""
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def problem_keywords(text: str) -> set[str]:
    """Extract meaningful keywords and a few explicit synonym expansions."""
    words = normalize(text).split()
    result: set[str] = set()

    for word in words:
        if len(word) <= 2 or word in STOPWORDS:
            continue
        result.add(word)
        for synonym in SYNONYM_MAP.get(word, "").split():
            result.add(synonym)

    return result


def text_from_node(node: Any) -> str:
    """Extract visible text from common YouTube JSON text nodes."""
    if isinstance(node, str):
        return node.strip()

    if isinstance(node, dict):
        if isinstance(node.get("simpleText"), str):
            return node["simpleText"].strip()

        runs = node.get("runs")
        if isinstance(runs, list):
            pieces = []
            for run in runs:
                if isinstance(run, dict) and isinstance(run.get("text"), str):
                    pieces.append(run["text"])
            if pieces:
                return "".join(pieces).strip()

    return ""


def first_text(node: Any, keys: tuple[str, ...]) -> str:
    """Find the first usable text value under any of the supplied keys."""
    if isinstance(node, dict):
        for key in keys:
            if key in node:
                value = text_from_node(node[key])
                if value:
                    return value

        for value in node.values():
            found = first_text(value, keys)
            if found:
                return found

    elif isinstance(node, list):
        for value in node:
            found = first_text(value, keys)
            if found:
                return found

    return ""


def first_video_id(node: Any) -> str:
    """Find a Shorts video ID in several generations of YouTube search JSON."""
    if isinstance(node, dict):
        # Current/recent Shorts navigation structure.
        reel_endpoint = node.get("reelWatchEndpoint")
        if isinstance(reel_endpoint, dict) and reel_endpoint.get("videoId"):
            return str(reel_endpoint["videoId"]).strip()

        # Older reelItemRenderer structure.
        if node.get("videoId") and (
            "reelItemRenderer" in str(node.get("navigationEndpoint", ""))
            or "headline" in node
            or "thumbnail" in node
        ):
            video_id = str(node["videoId"]).strip()
            if video_id:
                return video_id

        # Shorts lockup can contain an entity command with a reelWatchEndpoint.
        for key, value in node.items():
            if "shorts" in key.lower() and isinstance(value, (dict, list)):
                found = first_video_id(value)
                if found:
                    return found

        # Avoid treating every normal videoRenderer videoId as a Short.
        for value in node.values():
            found = first_video_id(value)
            if found:
                return found

    elif isinstance(node, list):
        for value in node:
            found = first_video_id(value)
            if found:
                return found

    return ""


def find_nodes_by_key(root: Any, wanted_keys: set[str]) -> Iterator[tuple[str, Any]]:
    """Recursively yield matching key/value pairs from YouTube's JSON."""
    if isinstance(root, dict):
        for key, value in root.items():
            if key in wanted_keys:
                yield key, value
            yield from find_nodes_by_key(value, wanted_keys)
    elif isinstance(root, list):
        for value in root:
            yield from find_nodes_by_key(value, wanted_keys)


def extract_initial_data(html: str) -> dict[str, Any]:
    """Extract ytInitialData JSON from the YouTube search HTML."""
    marker = re.search(r"(?:var\s+)?ytInitialData\s*=\s*", html)
    if not marker:
        raise ValueError("YouTube did not return ytInitialData (possible consent/bot page).")

    decoder = json.JSONDecoder()
    start = marker.end()
    try:
        data, _ = decoder.raw_decode(html[start:])
    except json.JSONDecodeError as exc:
        raise ValueError("Could not parse YouTube's search-page JSON.") from exc

    if not isinstance(data, dict):
        raise ValueError("Unexpected YouTube search data format.")
    return data


def parse_shorts_from_search(data: dict[str, Any], limit: int) -> list[dict[str, str]]:
    """Extract only explicit Shorts items from YouTube search-page data."""
    results: list[dict[str, str]] = []
    seen: set[str] = set()

    # YouTube has used both reelItemRenderer and Shorts lockup models over time.
    wanted = {"reelItemRenderer", "shortsLockupViewModel"}

    for key, item in find_nodes_by_key(data, wanted):
        if not isinstance(item, dict):
            continue

        if key == "reelItemRenderer":
            video_id = str(item.get("videoId") or "").strip()
        else:
            video_id = first_video_id(item)

        if not video_id or video_id in seen:
            continue

        # Only accept explicit Shorts structures. This prevents us from
        # fabricating /shorts/ URLs for ordinary videos.
        title = first_text(
            item,
            (
                "headline",
                "overlayMetadata",
                "overlayMetadataText",
                "title",
                "accessibilityText",
            ),
        )
        description = first_text(item, ("descriptionSnippet", "shortDescriptionText", "description"))
        channel = first_text(item, ("shortBylineText", "longBylineText", "ownerText", "channelName"))

        # Older Shorts often expose title in accessibility data.
        if not title:
            title = str(item.get("accessibilityText") or "YouTube Short").strip()

        seen.add(video_id)
        results.append(
            {
                "title": title or "YouTube Short",
                "description": description,
                "channel": channel or "Unknown",
                "video_id": video_id,
                "url": f"https://www.youtube.com/shorts/{video_id}",
            }
        )

        if len(results) >= limit:
            break

    return results


def search_youtube_shorts(query: str, limit: int = 10, timeout: int = 12) -> list[dict[str, str]]:
    """Scrape YouTube search HTML and return only explicit Shorts items."""
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    data = extract_initial_data(response.text)
    return parse_shorts_from_search(data, limit=limit)


def solution_signals(short: dict[str, str]) -> tuple[int, int]:
    """Return (helpful-signal count, unwanted-signal count)."""
    text = normalize(f"{short.get('title', '')} {short.get('description', '')}")
    words = set(text.split())
    return len(words & SOLUTION_TERMS), len(words & NEGATIVE_TERMS)


def relevance_score(user_problem: str, short: dict[str, str]) -> float:
    """Score problem relevance plus how clearly the Short helps solve it."""
    query_text = normalize(user_problem)
    query_words = problem_keywords(user_problem)
    title_text = normalize(short.get("title", ""))
    desc_text = normalize(short.get("description", ""))
    title_words = set(title_text.split())
    desc_words = set(desc_text.split())

    if not query_words:
        return 0.0

    title_overlap = len(query_words & title_words) / len(query_words)
    desc_overlap = len(query_words & desc_words) / len(query_words)
    fuzzy_title = SequenceMatcher(None, query_text, title_text).ratio()
    phrase_bonus = 0.20 if query_text and query_text in title_text else 0.0

    solution_count, negative_count = solution_signals(short)
    solution_bonus = min(solution_count / 5.0, 1.0) * 0.30
    negative_penalty = min(negative_count / 2.0, 1.0) * 0.30

    score = (
        0.40 * title_overlap
        + 0.20 * desc_overlap
        + 0.10 * fuzzy_title
        + phrase_bonus
        + solution_bonus
        - negative_penalty
    )
    return max(0.0, min(score, 1.0))


def is_solution_oriented(user_problem: str, short: dict[str, str]) -> bool:
    """
    Accept only Shorts that contain explicit help/action/recovery intent and
    still have a meaningful lexical connection to the user's problem.
    """
    combined = normalize(f"{short.get('title', '')} {short.get('description', '')}")
    solution_count, negative_count = solution_signals(short)

    action_phrases = (
        "how to", "what to do", "how do i", "how can i", "tips", "advice",
        "steps", "overcome", "deal with", "cope with", "recover from",
        "move on", "get over", "start again", "bounce back",
    )
    has_action_phrase = any(phrase in combined for phrase in action_phrases)
    helpful = has_action_phrase or solution_count >= 2

    words = set(combined.split())
    problem_words = problem_keywords(user_problem)
    problem_overlap = len(problem_words & words) / len(problem_words) if problem_words else 0.0

    return helpful and negative_count == 0 and problem_overlap >= 0.20


def build_search_queries(user_problem: str) -> list[str]:
    """Search the user's actual problem with explicit overcoming intent."""
    clean = re.sub(r"\s+", " ", user_problem).strip()
    queries = [
        f"{clean} how to overcome shorts in Hindi",
        f"{clean} what to do shorts in Hindi",
        f"{clean} how to deal with shorts in Hindi",
        f"{clean} advice tips shorts in Hindi",
        f"{clean} recovery coping shorts in Hindi",
    ]

    words = [w for w in normalize(clean).split() if w not in STOPWORDS and len(w) > 2]
    if words:
        base = " ".join(words)
        queries += [
            f"{base} how to overcome shorts in Hindi",
            f"{base} advice shorts in Hindi",
        ]

    expanded_words: list[str] = []
    for word in words:
        expanded_words.append(word)
        expanded_words.extend(SYNONYM_MAP.get(word, "").split())
    expanded = " ".join(dict.fromkeys(expanded_words))
    if expanded:
        queries.append(f"{expanded} overcome advice shorts in Hindi")

    unique: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = re.sub(r"\s+", " ", query).strip()
        key = normalize(query)
        if key and key not in seen:
            seen.add(key)
            unique.append(query)
    return unique


def suggest_shorts(problem_topic: str, count: int = MIN_SHORTS) -> list[dict[str, str]]:
    """Find and rank Shorts for an abstract topic without printing or saving chat data."""
    ranked: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for query in build_search_queries(problem_topic)[:3]:
        try:
            candidates = search_youtube_shorts(query, limit=SEARCH_RESULTS_PER_QUERY, timeout=12)
        except (requests.RequestException, ValueError) as exc:
            logger.warning("YouTube Shorts query failed (%s)", type(exc).__name__)
            continue
        except Exception as exc:
            logger.warning("YouTube Shorts query failed (%s)", type(exc).__name__)
            continue

        for short in candidates:
            video_id = str(short.get("video_id") or "")
            if not video_id or video_id in seen_ids or not is_solution_oriented(problem_topic, short):
                continue
            seen_ids.add(video_id)
            short["relevance_score"] = f"{relevance_score(problem_topic, short):.4f}"
            short["solution_oriented"] = True
            short["id"] = video_id
            ranked.append(short)

        if len(ranked) >= max(count * 3, count):
            break

    ranked.sort(key=lambda item: float(item["relevance_score"]), reverse=True)
    return ranked[:max(0, count)]


def find_relevant_shorts(user_problem: str, count: int = MIN_SHORTS) -> tuple[list[dict[str, str]], list[str]]:
    """Return only useful, solution-focused Shorts for the user's problem."""
    all_results: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    used_queries: list[str] = []

    for query in build_search_queries(user_problem):
        used_queries.append(query)
        print(f"\nSearching solution-focused YouTube Shorts: {query}")

        try:
            candidates = search_youtube_shorts(query, limit=SEARCH_RESULTS_PER_QUERY)
        except requests.RequestException as exc:
            print(f"  Network/search error: {exc}")
            continue
        except ValueError as exc:
            print(f"  YouTube parsing error: {exc}")
            continue
        except Exception as exc:
            print(f"  Unexpected search error: {exc}")
            continue

        accepted = 0
        for short in candidates:
            video_id = short["video_id"]
            if video_id in seen_ids:
                continue
            if not is_solution_oriented(user_problem, short):
                continue

            seen_ids.add(video_id)
            short["relevance_score"] = f"{relevance_score(user_problem, short):.4f}"
            short["solution_oriented"] = True
            short["matched_query"] = query
            all_results.append(short)
            accepted += 1

        print(f"  Accepted relevant/solution-focused Shorts: {accepted}")

        if len(all_results) >= max(count * 4, count):
            break

    all_results.sort(key=lambda x: float(x["relevance_score"]), reverse=True)
    return all_results[:count], used_queries

def save_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_results(user_problem: str, shorts: list[dict[str, str]]) -> None:
    print("\n" + "=" * 78)
    print("DIGITAL MENTOR - RELEVANT YOUTUBE SHORTS")
    print("=" * 78)
    print(f"Your problem: {user_problem}")

    if len(shorts) < MIN_SHORTS:
        print(f"\nOnly {len(shorts)} solution-focused relevant Short(s) were found.")
        print("No unrelated or fake /shorts/ links were generated.")
        print("The script will not fill the list with Shorts that merely mention the problem.")
        return

    print(f"\nFound {len(shorts)} solution-focused YouTube Shorts:\n")
    for index, short in enumerate(shorts, start=1):
        print(f"{index}. {short['title']}")
        print(f"   Channel   : {short['channel']}")
        print(f"   Relevance : {float(short['relevance_score']):.2f}")
        print(f"   URL       : {short['url']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find relevant YouTube Shorts for a user problem without a YouTube API key."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"JSON output file (default: {DEFAULT_OUTPUT.name})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=MIN_SHORTS,
        help=f"Number of Shorts to return (minimum {MIN_SHORTS}, default {MIN_SHORTS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count < MIN_SHORTS:
        raise SystemExit(f"--count must be at least {MIN_SHORTS}.")

    output_path = Path(args.output).expanduser().resolve()

    print("=" * 78)
    print("DIGITAL MENTOR - YOUTUBE SHORTS FINDER")
    print("=" * 78)
    print("No YouTube API key is required.")
    print("Describe the problem in your own words and I will search for relevant Shorts.")

    try:
        user_problem = input("\nYour problem: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        return

    if not user_problem:
        raise SystemExit("Please enter a problem description.")

    shorts, queries = find_relevant_shorts(user_problem, count=args.count)

    result = {
        "user_problem": user_problem,
        "required_shorts": args.count,
        "found_shorts": len(shorts),
        "status": "ok" if len(shorts) >= args.count else "partial",
        "search_queries": queries,
        "shorts": shorts,
        "api_key_used": False,
        "source": "YouTube public search HTML",
    }

    save_json(result, output_path)
    print_results(user_problem, shorts)

    print("=" * 78)
    print(f"Saved JSON: {output_path}")
    print("YouTube API key: NOT USED")
    print("Only Shorts explicitly exposed by YouTube's search data and passing the solution-focused relevance filter are returned.")
    print("=" * 78)


if __name__ == "__main__":
    main()
