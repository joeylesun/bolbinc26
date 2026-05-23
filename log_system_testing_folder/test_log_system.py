"""
test_log_system.py — verify the SUVOS UV-C logging system without hardware.
===========================================================================

Runs entirely on the standard library. No cameras, no YOLO, no OpenCV.
Your colleague can run this on any machine with Python — just hit Run.

It exercises the REAL logging core (suvos_zone_logic.ZoneLogger, the same
class suvos_working.py uses) by injecting scripted occupancy timelines
that stand in for what the camera + YOLO pipeline would produce. Each
scenario checks that the correct rows land in SQLite with the correct
reason strings.

What "mimicking working mode" means here:
  In production, every loop tick the system computes a per-zone occupancy
  dict (from YOLO detections + the darkness watchdog) and calls
  ZoneLogger.step(occupied, dt, any_dark). This test does exactly that,
  but the occupancy dict comes from a timeline we control instead of a
  camera. The FSM, transition detection, and SQLite writes are identical.
"""

import os
import sqlite3
import tempfile

from suvos_zone_logic import (
    ZoneLogger, State,
    MSG_ON_CLEANING, MSG_OFF_PERSON, MSG_OFF_STANDBY, MSG_OFF_DARKNESS,
)


# ─────────────────────────────────────────────
#  Tiny test framework (no pytest dependency)
# ─────────────────────────────────────────────
class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"    PASS  {name}")
        else:
            self.failed += 1
            print(f"    FAIL  {name}"
                  + (f"\n          {detail}" if detail else ""))

    def section(self, title):
        print(f"\n── {title} " + "─" * max(0, 60 - len(title)))

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 68)
        print(f"  RESULT: {self.passed}/{total} checks passed, "
              f"{self.failed} failed")
        print("=" * 68)
        return self.failed == 0


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
# Short FSM timings so scenarios run instantly (vs the 5/30/10s defaults)
FAST_TIMES = {
    "waitingTime":      2.0,
    "disinfectionTime": 3.0,
    "standbyTime":      2.0,
}


def fresh_db_path():
    """A clean temp DB file, removed if it already exists."""
    path = os.path.join(tempfile.gettempdir(), "suvos_test_log.db")
    if os.path.exists(path):
        os.remove(path)
    return path


def make_logger(zone_ids, times=None):
    return ZoneLogger(zone_ids, fresh_db_path(),
                      fsm_times=times or FAST_TIMES)


def simulate(logger, segments, dt=0.5):
    """Drive the logger through a scripted timeline.

    segments: list of (duration_seconds, occupied_zone_ids, any_dark).
      - occupied_zone_ids: an iterable of zone ids occupied during the
        segment (or 'ALL' to occupy every zone, used for darkness).
      - any_dark: if True, ALL zones are forced occupied (mirrors how the
        working loop forces occupancy when a camera goes dark) and the
        darkness flag is passed so the log reason is labelled correctly.
    """
    for duration, occ, any_dark in segments:
        steps = max(1, int(round(duration / dt)))
        for _ in range(steps):
            if any_dark or occ == "ALL":
                occupied = {z: True for z in logger.zone_ids}
            else:
                occ_set = set(occ)
                occupied = {z: (z in occ_set) for z in logger.zone_ids}
            logger.step(occupied, dt, any_dark=any_dark)


def statuses_for_zone(logger, zone_id):
    """Ordered list of status strings logged for one zone."""
    return [r[3] for r in logger.read_events() if r[2] == zone_id]


def all_statuses(logger):
    return [(r[2], r[3]) for r in logger.read_events()]


# ─────────────────────────────────────────────
#  Scenario tests
# ─────────────────────────────────────────────
def test_database_schema(r):
    r.section("Database initialization & schema")
    logger = make_logger([0])
    conn = sqlite3.connect(logger.db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(uvc_logs)")
    cols = {row[1]: row[2] for row in cur.fetchall()}
    conn.close()

    r.check("uvc_logs table exists with id column", "id" in cols)
    r.check("has timestamp column", "timestamp" in cols)
    r.check("has zone_id column", "zone_id" in cols)
    r.check("has status column", "status" in cols)
    r.check("fresh DB starts empty", len(logger.read_events()) == 0)


def test_log_event_write_read(r):
    r.section("Direct log_event write / read round-trip")
    logger = make_logger([0])
    logger.log_event(0, "TEST MESSAGE")
    logger.log_event(3, "ANOTHER")
    rows = logger.read_events()
    r.check("two rows written", len(rows) == 2)
    r.check("first row zone_id correct", rows[0][2] == 0)
    r.check("first row status correct", rows[0][3] == "TEST MESSAGE")
    r.check("second row zone_id correct", rows[1][2] == 3)
    r.check("timestamp is populated", bool(rows[0][1]))


def test_clean_cycle_starts(r):
    r.section("Empty room → CLEAN starts → 'ON (CLEANING)'")
    logger = make_logger([0])
    # Empty for longer than waitingTime (2s), should trigger CLEAN
    simulate(logger, [(4.0, [], False)])
    s = statuses_for_zone(logger, 0)
    r.check("exactly one ON logged", s.count(MSG_ON_CLEANING) == 1,
            f"got {s}")
    r.check("first event is ON (CLEANING)",
            len(s) >= 1 and s[0] == MSG_ON_CLEANING, f"got {s}")


def test_person_interrupts_clean(r):
    r.section("Person enters during CLEAN → 'OFF (PERSON DETECTED)'")
    logger = make_logger([0])
    simulate(logger, [
        (4.0, [],  False),   # empty → CLEAN turns ON
        (2.0, [0], False),   # person in zone 0 → UV-C must turn OFF
    ])
    s = statuses_for_zone(logger, 0)
    r.check("logged ON then OFF(PERSON)",
            s == [MSG_ON_CLEANING, MSG_OFF_PERSON], f"got {s}")


def test_clean_completes_standby(r):
    r.section("CLEAN runs to completion → 'OFF (STANDBY TIMEOUT)'")
    logger = make_logger([0])
    # Empty the whole time: CLEAN starts, runs full disinfectionTime (3s),
    # then transitions to STBY (bit 0) with nobody present.
    simulate(logger, [(8.0, [], False)])
    s = statuses_for_zone(logger, 0)
    r.check("logged ON then OFF(STANDBY)",
            s[:2] == [MSG_ON_CLEANING, MSG_OFF_STANDBY], f"got {s}")
    r.check("OFF reason is STANDBY not PERSON",
            MSG_OFF_PERSON not in s, f"got {s}")


def test_darkness_watchdog(r):
    r.section("Lights out during CLEAN → 'OFF (DARKNESS WATCHDOG)'")
    logger = make_logger([0])
    simulate(logger, [
        (4.0, [],    False),   # empty → CLEAN ON
        (2.0, "ALL", True),    # darkness: all zones forced occupied + flag
    ])
    s = statuses_for_zone(logger, 0)
    r.check("logged ON then OFF(DARKNESS)",
            s == [MSG_ON_CLEANING, MSG_OFF_DARKNESS], f"got {s}")
    r.check("darkness reason beats person reason",
            MSG_OFF_PERSON not in s, f"got {s}")


def test_multi_zone_independence(r):
    r.section("Two zones cycle independently")
    logger = make_logger([0, 1])
    # Zone 0 stays empty (will clean); zone 1 has a person the whole time
    # (must NEVER turn on).
    simulate(logger, [(8.0, [1], False)])
    s0 = statuses_for_zone(logger, 0)
    s1 = statuses_for_zone(logger, 1)
    r.check("zone 0 ran a cycle (has ON)", MSG_ON_CLEANING in s0,
            f"zone0={s0}")
    r.check("zone 1 never turned ON (occupied throughout)",
            MSG_ON_CLEANING not in s1, f"zone1={s1}")
    r.check("zone 1 produced no log rows at all", len(s1) == 0,
            f"zone1={s1}")


def test_no_spurious_logs_when_idle(r):
    r.section("No transitions logged while state is steady")
    logger = make_logger([0])
    # Person present the entire time → zone never leaves WAIT, bit never
    # changes, so zero log rows should be written.
    simulate(logger, [(6.0, [0], False)])
    r.check("no rows logged when always occupied",
            len(logger.read_events()) == 0,
            f"got {all_statuses(logger)}")


def test_repeated_cycles(r):
    r.section("Multiple full cycles produce repeated ON/OFF pairs")
    logger = make_logger([0])
    # Two empty windows separated by a person, forcing two CLEAN cycles.
    simulate(logger, [
        (8.0, [],  False),   # cycle 1: ON then OFF(STANDBY)
        (1.0, [0], False),   # person resets to WAIT
        (8.0, [],  False),   # cycle 2: ON then OFF(STANDBY)
    ])
    s = statuses_for_zone(logger, 0)
    r.check("at least two ON events", s.count(MSG_ON_CLEANING) >= 2,
            f"got {s}")


# ─────────────────────────────────────────────
#  Realistic working-mode simulation (for eyeballing)
# ─────────────────────────────────────────────
def demo_realistic_timeline():
    print("\n" + "=" * 68)
    print("  REALISTIC WORKING-MODE SIMULATION (3 zones)")
    print("=" * 68)
    print("  Mimics a real shift: rooms empty out, get cleaned, people")
    print("  wander through, and the lights cut out once mid-shift.\n")

    logger = make_logger([0, 1, 2], times={
        "waitingTime": 3.0, "disinfectionTime": 5.0, "standbyTime": 2.0})

    timeline = [
        #  dur   occupied      dark   description
        (4.0,  [0, 1, 2], False),   # everyone present, nothing cleans
        (8.0,  [],        False),   # everyone leaves → all 3 zones clean
        (2.0,  [1],       False),   # someone walks into zone 1
        (6.0,  [],        False),   # they leave → zone 1 cleans again
        (3.0,  "ALL",     True),    # LIGHTS OUT — darkness watchdog fires
        (6.0,  [],        False),   # lights back, rooms clean again
    ]
    simulate(logger, timeline, dt=0.5)

    rows = logger.read_events()
    print(f"  {'#':>3}  {'zone':>4}  {'status'}")
    print("  " + "-" * 50)
    for row in rows:
        _id, _ts, zid, status = row
        print(f"  {_id:>3}  {zid:>4}  {status}")
    print(f"\n  Total log rows written: {len(rows)}")
    print("  (Each ON should be followed by an OFF with a sensible reason.)")


# ─────────────────────────────────────────────
#  Run everything
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 68)
    print("  SUVOS LOG SYSTEM TEST  (no cameras / no YOLO required)")
    print("=" * 68)

    r = Runner()
    test_database_schema(r)
    test_log_event_write_read(r)
    test_clean_cycle_starts(r)
    test_person_interrupts_clean(r)
    test_clean_completes_standby(r)
    test_darkness_watchdog(r)
    test_multi_zone_independence(r)
    test_no_spurious_logs_when_idle(r)
    test_repeated_cycles(r)

    ok = r.summary()

    # Always show the human-readable realistic run at the end
    demo_realistic_timeline()

    print("\nDone." + ("  All automated checks passed." if ok
                        else "  SOME CHECKS FAILED — see above."))
