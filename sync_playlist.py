#!/usr/bin/env python3
"""
sync_playlist.py

Pulls track URLs from a Spotify playlist, compares them against a local
state file of previously-seen URLs, and downloads only the NEW ones with
spotdl.

Each playlist automatically gets its own folder under MUSIC_BASE_DIR, named
after the playlist (sanitized for the filesystem). That folder is where
tracks get downloaded to, and it also holds the per-playlist state file
used for the "only download new tracks" check, so multiple playlists never
share state and never collide with each other's downloads.

Setup:
    pip install spotipy

    Create a free app at https://developer.spotify.com/dashboard
    In the app's settings, add this exact Redirect URI: http://127.0.0.1:8888/callback
    Then set these env vars (or hardcode them below):

        export SPOTIPY_CLIENT_ID="your_client_id"
        export SPOTIPY_CLIENT_SECRET="your_client_secret"

    First run will open a browser to log in and approve access. After that,
    a cached token (.cache file) means you won't need to log in again.

    YouTube now blocks yt-dlp downloads ("Sign in to confirm you're not a
    bot") without both of the following:

      1. Cookies from a real logged-in-to-YouTube browser session (see
         YT_DLP_ARGS below — defaults to reading them from Firefox, so make
         sure you're logged into youtube.com there).

      2. A local PO token provider, so yt-dlp can pass YouTube's BotGuard
         origin check. One-time setup:

            pip install bgutil-ytdlp-pot-provider
            git clone --depth 1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git ~/bgutil-ytdlp-pot-provider
            cd ~/bgutil-ytdlp-pot-provider/server && deno install

         yt-dlp auto-detects it from that default path — no extra flags or
         code changes needed once it's there. Requires deno >= 2.4.3 (or
         node >= 22) on PATH.

Usage:
    python3 sync_playlist.py <playlist_url_or_id> [--state state.json] [--dry-run]

Example:
    python3 sync_playlist.py https://open.spotify.com/playlist/xxxxxxxx
    python3 sync_playlist.py xxxxxxxx --state my_playlist_state.json
"""

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from rich.console import Console, Group
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

# Parent directory that per-playlist folders get created under.
# Each playlist gets its own "<MUSIC_BASE_DIR>/<Playlist Name>/" folder,
# auto-created on first run. Override per-run with --output if you want a
# specific location instead.
MUSIC_BASE_DIR = "/home/Penjinear/Music"

# Name of the per-playlist state file created inside each playlist's folder.
STATE_FILENAME = ".playlist_state.json"

# Highest quality mp3 encoding. Override per-run with --bitrate if needed.
DEFAULT_BITRATE = "320k"
DEFAULT_FORMAT = "mp3"

# How many tracks to download concurrently. Most of the per-track time is
# spent waiting on YouTube's search API (which retries with backoff), so
# running several downloads in parallel overlaps that wait instead of
# fixing it. Diminishing returns past ~4-6. Override with --threads.
DEFAULT_THREADS = 4

# YouTube now blocks yt-dlp's requests outright ("Sign in to confirm you're
# not a bot") unless the request carries cookies from a real logged-in
# browser session. --cookies-from-browser reads those cookies directly from
# the named browser's local profile. Also requires a JS runtime (e.g. deno)
# on PATH for yt-dlp to solve YouTube's signature challenge at all, and the
# bgutil PO token provider set up per the instructions above.
#
# Deliberately NOT forcing --extractor-args youtube:player_client=... here:
# an earlier version of this pinned player_client=web_embedded (needed
# before cookies/PO tokens were set up), but web_embedded specifically
# fails ("Video unavailable"/UNPLAYABLE) on any video where the uploader
# disabled outside embedding — common for official/label music content.
# Confirmed 2026-09-04: with cookies + PO token now in place, leaving the
# player client unpinned lets yt-dlp fall back across clients on its own
# and succeeds on videos that used to fail this way.
#
# Chromium-based browsers (chrome, edge, etc.) lock their cookie database
# while running, so yt-dlp can fail to read it unless that browser is
# closed first. Firefox doesn't have this problem, hence the default.
YT_DLP_ARGS = "--cookies-from-browser firefox"

# Must exactly match a Redirect URI registered in your Spotify app settings.
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "playlist-read-private playlist-read-collaborative"

# Characters that are illegal or awkward in filesystem folder names.
_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def extract_playlist_id(url_or_id: str) -> str:
    """Accepts a full Spotify playlist URL, a URI, or a bare ID."""
    match = re.search(r"playlist[/:]([a-zA-Z0-9]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id  # assume it's already a bare ID


def sanitize_folder_name(name: str) -> str:
    """Turns a playlist name into a safe, single folder-name component."""
    sanitized = _UNSAFE_NAME_CHARS.sub("_", name)
    sanitized = re.sub(r"\s+", " ", sanitized)
    sanitized = sanitized.strip(" .")
    return sanitized or "Untitled Playlist"


def get_playlist_name(sp: spotipy.Spotify, playlist_id: str) -> str:
    info = sp.playlist(playlist_id, fields="name")
    return info.get("name") or "Untitled Playlist"


def get_all_playlist_urls(
    sp: spotipy.Spotify, playlist_id: str, market: str
) -> tuple[list[str], list[str], dict[str, str]]:
    urls = []
    skipped_names = []
    track_names: dict[str, str] = {}
    results = sp.playlist_items(playlist_id, additional_types=["track"], market=market)
    while results:
        for entry in results["items"]:
            # Spotify's response nests track/episode data under "item" (previously "track").
            track = entry.get("item") or entry.get("track")
            if track and track.get("type") == "track" and track.get("external_urls", {}).get("spotify"):
                url = track["external_urls"]["spotify"]
                urls.append(url)
                artists = ", ".join(a.get("name", "?") for a in track.get("artists", []))
                name = track.get("name", "Unknown title")
                track_names[url] = f"{artists} - {name}" if artists else name
            else:
                if track:
                    artists = ", ".join(a.get("name", "?") for a in track.get("artists", []))
                    name = track.get("name", "Unknown title")
                    label = f"{name} — {artists}" if artists else name
                elif entry.get("is_local"):
                    label = "Local file (no Spotify data)"
                else:
                    label = "Unknown item (no data returned)"
                skipped_names.append(label)
        results = sp.next(results) if results.get("next") else None
    return urls, skipped_names, track_names


def load_state(state_path: Path) -> set[str]:
    if not state_path.exists():
        return set()
    try:
        with open(state_path, "r") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"Warning: state file {state_path} could not be read ({e}); "
            "treating it as empty (all current tracks will look 'new').",
            file=sys.stderr,
        )
        return set()


def save_state(state_path: Path, urls: set[str]) -> None:
    # Write to a temp file and atomically replace, so a crash mid-write
    # can't leave behind a truncated/corrupted state file.
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(sorted(urls), f, indent=2)
    os.replace(tmp_path, state_path)


def reconciled_state(previous_urls: set[str], current_urls: set[str], newly_succeeded: set[str] = frozenset()) -> set[str]:
    """Tracks currently aren't tracked (removed from the playlist) or successfully downloaded (kept)."""
    return (previous_urls & current_urls) | set(newly_succeeded)


# spotdl (in --simple-tui mode) logs one line per download-phase change, in
# the form "<song name>: <phase>". Rich's normal spinner/percentage bar
# never emits anything meaningful when spotdl's stdout isn't a real
# terminal (it only draws in place via cursor control codes), so this is
# the only real progress signal available from a piped subprocess — these
# phases and percentages mirror the ones spotdl's own ProgressHandler uses
# internally (spotdl/download/progress_handler.py).
PHASE_PROGRESS = {
    "Searching for song": 25,
    "Getting audio meta": 40,
    "Downloading": 70,
    "Converting": 70,
    "Embedding metadata": 95,
    "Done": 100,
    "Skipped": 100,
    "Error": 100,
}

_thread_slots = threading.local()
_slot_lock = threading.Lock()
_next_slot = [0]


def _slot_for_current_thread() -> int:
    """Assigns each ThreadPoolExecutor worker thread a stable display row
    (0..threads-1) the first time it picks up a job, and reuses that same
    row for every job that thread runs afterward. ThreadPoolExecutor reuses
    its worker threads across submitted tasks, so this gives a fixed set of
    rows instead of one row per track (which is what caused the constantly
    changing/scrolling names)."""
    if not hasattr(_thread_slots, "id"):
        with _slot_lock:
            _thread_slots.id = _next_slot[0]
            _next_slot[0] += 1
    return _thread_slots.id


def download_one(
    url: str,
    display_name: str,
    output_dir: Path,
    output_template_name: str,
    bitrate: str,
    audio_format: str,
    progress: Progress,
    slot_tasks: list,
    proc_registry: dict,
    proc_lock: threading.Lock,
    stop_event: threading.Event,
) -> tuple[str, bool]:
    """Downloads a single track into its own private temp folder (so
    concurrent downloads can never be confused with each other), then moves
    the result into output_dir. spotdl/yt-dlp can exit 0 even when the
    actual audio download failed (e.g. YouTube 403 after a failed signature
    challenge), so a zero return code alone isn't proof a file was made —
    only a file actually appearing counts as success."""
    if stop_event.is_set():
        # A worker thread can free up (its previous track got killed by an
        # interrupt) and pull the next queued track off the pool's internal
        # queue faster than the interrupt handler can cancel it — this check
        # is what stops that thread from starting a brand new, untracked
        # subprocess that the interrupt handler would never know to kill.
        return url, False

    slot = _slot_for_current_thread()
    task_id = slot_tasks[slot]
    # reset (not update): each slot's row is reused across many tracks, and
    # TimeElapsedColumn reports time since the task's start_time — without
    # resetting it here, the timer would keep counting from when the row
    # was first created (i.e. since the whole batch started) instead of
    # since this particular track began.
    progress.reset(task_id, description=f"Downloading: {display_name}", completed=0)

    # No leading dot in the prefix: spotdl/yt-dlp silently strips leading
    # dots from output-path directory components (to avoid creating hidden
    # folders), which would make it write into a different, non-hidden
    # sibling directory than the one we actually created and are watching.
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="dl_tmp_") as tmp_dir:
        cmd = [
            "spotdl", "download", url,
            "--bitrate", bitrate,
            "--format", audio_format,
            "--yt-dlp-args", YT_DLP_ARGS,
            # spotdl only searches YouTube Music's curated catalog by
            # default, which doesn't have an entry for every track (niche/
            # indie releases especially) and fails with "no usable results"
            # even though a plain YouTube search would find something.
            # Falls back to a plain YouTube search when youtube-music comes
            # up empty.
            "--audio", "youtube-music", "youtube",
            "--output", str(Path(tmp_dir) / output_template_name),
            # Without this, spotdl detects its stdout isn't a real terminal
            # and switches to drawing its progress bar with cursor-control
            # codes that only make sense on a live TTY — piped through a
            # subprocess that's just inert, invisible bytes. --simple-tui
            # instead logs one plain text line per phase change ("<song>:
            # Searching for song" / "...: Downloading" / etc.), which is
            # what actually lets us drive our own progress display below.
            "--simple-tui",
        ]
        # Output is captured (not streamed to the terminal) so that up to
        # `threads` concurrent spotdl processes can't interleave/clobber
        # each other's lines on screen. Each line is parsed for spotdl's
        # phase markers to advance this track's row in real time; the full
        # output is only printed if the download fails, to help debug why.
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with proc_lock:
            proc_registry[url] = proc
        output_lines = []
        try:
            for line in proc.stdout:
                output_lines.append(line)
                stripped = line.strip()
                for phase, pct in PHASE_PROGRESS.items():
                    if stripped.endswith(phase):
                        progress.update(task_id, description=f"{phase}: {display_name}", completed=pct)
                        break
            proc.wait()
        finally:
            with proc_lock:
                proc_registry.pop(url, None)

        downloaded = list(Path(tmp_dir).glob(f"*.{audio_format}"))
        ok = proc.returncode == 0 and bool(downloaded)

        if ok:
            shutil.move(str(downloaded[0]), str(output_dir / downloaded[0].name))
            progress.update(task_id, description=f"Done: {display_name}", completed=100)
        else:
            progress.update(task_id, description=f"Failed: {display_name}", completed=100)
            if proc.returncode == 0 and not downloaded:
                reason = f"spotdl reported success but no .{audio_format} file was produced"
            else:
                # spotdl's own error/traceback text is almost always the
                # last non-blank line it printed before exiting — pulling
                # just that line out (instead of dumping every captured
                # phase-progress line) is what actually shows *why* it
                # failed instead of burying it in noise.
                reason = next((l.strip() for l in reversed(output_lines) if l.strip()), "no output captured")
            progress.console.print(f"\n[red]Failed:[/red] {display_name}")
            progress.console.print(f"  {url}")
            progress.console.print(f"  [red]Reason:[/red] {reason}")

        return url, ok


def main():
    parser = argparse.ArgumentParser(description="Sync new tracks from a Spotify playlist via spotdl")
    parser.add_argument("playlist", help="Spotify playlist URL, URI, or ID")
    parser.add_argument(
        "--state",
        default=None,
        help="Path to state file (default: <playlist folder>/" + STATE_FILENAME + ")",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show new URLs without downloading")
    parser.add_argument("--debug", action="store_true", help="Print raw JSON for the first few playlist items, then exit")
    parser.add_argument(
        "--output",
        default=None,
        help="Passed through to spotdl --output (default: <playlist folder>/{title}.{output-ext})",
    )
    parser.add_argument(
        "--bitrate",
        default=DEFAULT_BITRATE,
        help=f"Passed through to spotdl --bitrate (default: {DEFAULT_BITRATE})",
    )
    parser.add_argument(
        "--format",
        dest="audio_format",
        default=DEFAULT_FORMAT,
        help=f"Passed through to spotdl --format (default: {DEFAULT_FORMAT})",
    )
    parser.add_argument(
        "--market",
        default="GB",
        help="Spotify market/country code used when fetching playlist items (default: GB)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Number of tracks to download concurrently (default: {DEFAULT_THREADS})",
    )
    args = parser.parse_args()

    client_id = os.environ.get("SPOTIPY_CLIENT_ID")
    client_secret = os.environ.get("SPOTIPY_CLIENT_SECRET")
    if not client_id or not client_secret:
        sys.exit(
            "Error: set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET env vars "
            "(get them from https://developer.spotify.com/dashboard)"
        )

    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotify_auth_cache",
    ))

    playlist_id = extract_playlist_id(args.playlist)

    try:
        if args.debug:
            print(f"Fetching raw playlist data for: {playlist_id}\n")
            raw = sp.playlist_items(playlist_id, additional_types=["track"], market=args.market, limit=3)
            print(json.dumps(raw, indent=2))
            return

        playlist_name = get_playlist_name(sp, playlist_id)
    except spotipy.SpotifyException as e:
        sys.exit(f"Error: Spotify API request failed ({e}). Check that the playlist ID/URL is correct and accessible.")

    folder_name = sanitize_folder_name(playlist_name)
    playlist_dir = Path(MUSIC_BASE_DIR) / folder_name
    folder_already_existed = playlist_dir.exists()

    if folder_already_existed:
        print(f"Found existing folder for playlist \"{playlist_name}\": {playlist_dir}")
    elif args.dry_run:
        print(f"No folder found for playlist \"{playlist_name}\" — would create folder: {playlist_dir}")
    else:
        playlist_dir.mkdir(parents=True, exist_ok=True)
        print(f"No folder found for playlist \"{playlist_name}\" — created new folder: {playlist_dir}")

    if args.state is None:
        args.state = str(playlist_dir / STATE_FILENAME)
    if args.output is None:
        args.output = str(playlist_dir / "{title}.{output-ext}")

    print(f"Fetching tracks for playlist: {playlist_id}")
    try:
        found_urls, skipped_names, track_names = get_all_playlist_urls(sp, playlist_id, args.market)
    except spotipy.SpotifyException as e:
        sys.exit(f"Error: Spotify API request failed ({e}). Check that the playlist ID/URL is correct and accessible.")

    current_urls = set(found_urls)
    print(f"Found {len(current_urls)} tracks currently in playlist.")

    if skipped_names:
        print(f"\n{len(skipped_names)} item(s) could not be matched to a downloadable Spotify track:")
        for name in skipped_names:
            print(f"  - {name}")
        print()

    state_path = Path(args.state)
    previous_urls = load_state(state_path)
    new_urls = current_urls - previous_urls

    if not new_urls:
        print("No new tracks since last run.")
        save_state(state_path, reconciled_state(previous_urls, current_urls))
        return

    print(f"Found {len(new_urls)} new track(s):")
    for url in sorted(new_urls):
        print(f"  {url}")

    if args.dry_run:
        print("\n--dry-run set, not downloading and not updating state file.")
        return

    if shutil.which("spotdl") is None:
        sys.exit("Error: 'spotdl' not found on PATH. Install it with: pip install spotdl")

    if not any(shutil.which(rt) for rt in ("deno", "node", "bun", "quickjs")):
        print(
            "Warning: no JS runtime (deno/node/bun/quickjs) found on PATH. yt-dlp needs one "
            "to solve YouTube's signature challenge — downloads will likely fail with HTTP 403.",
            file=sys.stderr,
        )

    if not (Path.home() / "bgutil-ytdlp-pot-provider" / "server" / "src" / "generate_once.ts").exists():
        print(
            "Warning: bgutil PO token provider not found at ~/bgutil-ytdlp-pot-provider/server. "
            "Without it, YouTube will likely block downloads with 'Sign in to confirm you're not "
            "a bot'. See the setup instructions at the top of this file.",
            file=sys.stderr,
        )

    # Make sure the destination folder exists (spotdl won't create nested dirs itself)
    output_dir = Path(args.output).parent
    output_template_name = Path(args.output).name
    output_dir.mkdir(parents=True, exist_ok=True)

    threads = max(1, args.threads)
    print(f"\nDownloading {len(new_urls)} track(s) with {threads} concurrent worker(s)...")

    succeeded = set()
    failed = []
    state_lock = threading.Lock()
    proc_registry: dict[str, subprocess.Popen] = {}
    proc_lock = threading.Lock()
    stop_event = threading.Event()

    # Shared so stray progress.console.print() calls (e.g. on a failed
    # track) cooperate with Live instead of corrupting its render — both
    # progress bars and the Live instance below all need to point at the
    # exact same Console object for that to work.
    console = Console()

    overall_progress = Progress(
        TextColumn("Downloaded"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("tracks"),
        TextColumn("[red]({task.fields[failed]} failed)[/red]"),
        console=console,
    )
    overall_task_id = overall_progress.add_task("overall", total=len(new_urls), failed=0)

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )
    # One fixed row per worker thread, reused for every track that thread
    # downloads, instead of a new line per track.
    slot_tasks = [progress.add_task("Waiting...", total=100) for _ in range(threads)]

    display = Group(overall_progress, progress)

    # Tracks URLs already folded into succeeded/failed, so a track can't be
    # recorded twice — needed because the KeyboardInterrupt handler below
    # re-checks every future (including ones the main loop already
    # consumed) to catch any that finished without being processed yet.
    processed_urls: set[str] = set()

    def record_result(url: str, ok: bool) -> None:
        with state_lock:
            if url in processed_urls:
                return
            processed_urls.add(url)
            if ok:
                succeeded.add(url)
                # Save immediately after each success so a later crash/interrupt
                # can't lose progress already made.
                save_state(state_path, reconciled_state(previous_urls, current_urls, succeeded))
            else:
                failed.append(url)
            overall_progress.update(overall_task_id, advance=1, failed=len(failed))

    futures = []
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
    try:
        with Live(display, console=console, refresh_per_second=10):
            futures = [
                executor.submit(
                    download_one, url, track_names.get(url, url), output_dir, output_template_name,
                    args.bitrate, args.audio_format, progress, slot_tasks,
                    proc_registry, proc_lock, stop_event,
                )
                for url in sorted(new_urls)
            ]
            for future in concurrent.futures.as_completed(futures):
                url, ok = future.result()
                record_result(url, ok)
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        # Without this, Ctrl+C only interrupts the main thread: worker
        # threads keep blocking on their spotdl subprocess, and at
        # interpreter exit Python unconditionally joins every thread any
        # ThreadPoolExecutor ever created — so the process won't actually
        # end until every worker thread's current call returns, no matter
        # what shutdown() is told.
        print("\n\nInterrupted — stopping in-flight downloads...", file=sys.stderr)

        try:
            # Set this before anything else: it stops any worker thread that
            # frees up (because we killed its subprocess below) from grabbing
            # the next queued track and starting a fresh, untracked subprocess
            # while we're still in the middle of shutting down.
            stop_event.set()
            # Drop everything not yet started.
            executor.shutdown(wait=False, cancel_futures=True)

            # Kill whatever's currently in flight. Swept a few times: a worker
            # can still win a narrow race and start exactly one more subprocess
            # right at the boundary above, so keep sweeping until the registry
            # is empty rather than assuming one pass caught everything.
            for _ in range(20):
                with proc_lock:
                    procs = list(proc_registry.values())
                if not procs:
                    break
                for proc in procs:
                    proc.terminate()
                for proc in procs:
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            # A track can fully succeed (file written, spotdl process exited 0)
            # in the moment between it finishing and the main loop's
            # as_completed() getting around to consuming its future — if the
            # interrupt lands in exactly that window, that success would
            # otherwise never make it into succeeded/the state file, even
            # though the file already exists on disk. That'd make the next run
            # redownload something that already downloaded fine. So: collect
            # the result of every future that actually got to run.
            #
            # This blocks rather than just checking future.done(): every
            # subprocess has already been killed and waited on above, but the
            # worker thread that owned it still needs a moment to do its
            # trailing bookkeeping (move the file, clean up its temp dir,
            # return) — future.done() can read False in that brief window and
            # skip a real success. Since there's no more process I/O left for
            # it to block on, .result() resolves almost immediately; the
            # timeout is just a safety net against something unforeseen
            # hanging forever.
            for future in futures:
                if future.cancelled():
                    continue
                try:
                    url, ok = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    continue
                record_result(url, ok)
        except KeyboardInterrupt:
            # A second Ctrl+C landing mid-cleanup — someone who already
            # asked once to stop is not going to want to wait through a
            # careful shutdown a second time. Skip straight to killing
            # every process we know about with SIGKILL (no more waiting
            # for a clean exit) and bail out via os._exit(), which ends
            # the process immediately and skips Python's normal shutdown
            # (which is exactly what would otherwise hang here again,
            # waiting to join worker threads still stuck on a subprocess).
            # Trade-off: any success still in flight at this exact moment
            # won't get recorded and will simply redownload next run.
            print("\nSecond interrupt — killing everything immediately.", file=sys.stderr)
            with proc_lock:
                procs = list(proc_registry.values())
            for proc in procs:
                proc.kill()
            os._exit(130)

        if succeeded:
            print(f"{len(succeeded)} track(s) downloaded before interrupt. State file updated.", file=sys.stderr)
        sys.exit(130)

    if succeeded:
        print(f"\n{len(succeeded)} track(s) downloaded successfully. State file updated.")

    if failed:
        print(f"\n{len(failed)} track(s) FAILED to download (will retry next run):")
        for url in failed:
            print(f"  {track_names.get(url, url)}")
            print(f"    {url}")
        sys.exit(1)
    else:
        print("\nAll new tracks downloaded successfully.")


if __name__ == "__main__":
    main()
