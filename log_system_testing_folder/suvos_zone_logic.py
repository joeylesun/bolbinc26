"""
suvos_zone_logic.py — camera-independent zone FSM + UV-C event logging.

This is the logging/state-machine core of SUVOS, pulled out of
suvos_working.py so it can be exercised WITHOUT cameras, YOLO, OpenCV,
serial, or websockets. It depends only on the Python standard library
(sqlite3, datetime), so it can be tested on any machine.

suvos_working.py should import State, LedFSM, and ZoneLogger from here
rather than redefining them, so the production code and the test exercise
exactly the same logic.

The contract:
  • You feed step() a per-zone occupancy dict and a dt (seconds elapsed).
  • step() advances each zone's state machine, computes the UV-C bit
    vector, detects transitions, and writes a log row for each change.
  • Occupancy is whatever upstream produced it — YOLO detections in
    production, or a scripted timeline in the test. The logger doesn't
    care where it came from.
"""

import os
import sqlite3
from datetime import datetime


NUM_LEDS = 16

DEFAULT_FSM_TIMES = {
    "waitingTime":      5.0,   # empty-room time before CLEAN starts
    "disinfectionTime": 30.0,  # active UV-C duration
    "standbyTime":      10.0,  # cooldown before returning to WAIT
}

# Log message strings — single source of truth so tests can assert on them
MSG_ON_CLEANING   = "ON (CLEANING)"
MSG_OFF_PERSON    = "OFF (PERSON DETECTED)"
MSG_OFF_STANDBY   = "OFF (STANDBY TIMEOUT)"
MSG_OFF_DARKNESS  = "OFF (DARKNESS WATCHDOG)"


class State:
    WAIT = 0
    CLEAN = 1
    STBY = 2

    @staticmethod
    def name(s):
        return {0: "WAIT", 1: "CLEAN", 2: "STBY"}.get(s, "?")


class LedFSM:
    """Per-zone three-state machine. Identical logic to the original
    _LedFSM in suvos_working.py, but takes its timing dict by reference
    so the test can use short durations."""

    def __init__(self, fsm_times):
        self.state = State.WAIT
        self.timer = 0.0
        self.fsm_times = fsm_times

    def update(self, person_detected, dt):
        cfg = self.fsm_times
        if self.state == State.WAIT:
            if person_detected:
                self.timer = 0.0
            else:
                self.timer = min(self.timer + dt, cfg["waitingTime"])
                if self.timer >= cfg["waitingTime"]:
                    self.state = State.CLEAN
                    self.timer = 0
        elif self.state == State.CLEAN:
            if person_detected:
                self.state = State.WAIT
                self.timer = 0
            else:
                self.timer = min(self.timer + dt, cfg["disinfectionTime"])
                if self.timer >= cfg["disinfectionTime"]:
                    self.state = State.STBY
                    self.timer = 0
        elif self.state == State.STBY:
            if person_detected:
                self.state = State.WAIT
                self.timer = 0
            else:
                self.timer = min(self.timer + dt, cfg["standbyTime"])
                if self.timer >= cfg["standbyTime"]:
                    self.state = State.WAIT
        return self.state


class ZoneLogger:
    """Owns the per-zone FSMs, computes the UV-C bit vector, and logs
    every transition to SQLite with a reason. Camera-independent."""

    def __init__(self, zone_ids, db_path, fsm_times=None, num_leds=NUM_LEDS):
        self.zone_ids = list(zone_ids)
        self.db_path = db_path
        self.fsm_times = fsm_times if fsm_times else dict(DEFAULT_FSM_TIMES)
        self.num_leds = num_leds
        self.fsms = {zid: LedFSM(self.fsm_times) for zid in self.zone_ids}
        self.last_bits = tuple()
        self._init_database()

    # ── SQLite ──
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS uvc_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                zone_id INTEGER,
                status TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def log_event(self, zone_id, status):
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cur.execute(
                'INSERT INTO uvc_logs (timestamp, zone_id, status) '
                'VALUES (?, ?, ?)', (t, zone_id, status))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ZoneLogger] DB write failed: {e}")

    def read_events(self):
        """Return all logged rows as a list of (id, timestamp, zone_id,
        status) tuples, ordered chronologically."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, timestamp, zone_id, status FROM uvc_logs ORDER BY id")
        rows = cur.fetchall()
        conn.close()
        return rows

    # ── The one method working mode calls every loop tick ──
    def step(self, occupied, dt, any_dark=False):
        """Advance all zones one tick.

        occupied: dict {zone_id: bool}. In production this combines YOLO
                  detection TTL with the darkness watchdog. In the test it's
                  scripted.
        dt:       seconds elapsed since the previous tick.
        any_dark: True if the brightness watchdog tripped on any camera.
                  Used only to label the log reason; the caller is
                  responsible for having already forced occupancy True when
                  dark (mirrors suvos_working.py).

        Returns (bits, zone_states) where bits is the UV-C output vector and
        zone_states is a list of per-zone dicts for the UI/WebSocket layer.
        """
        bits = [0] * self.num_leds
        zone_states = []

        for zid in self.zone_ids:
            fsm = self.fsms[zid]
            occ = bool(occupied.get(zid, False))
            state = fsm.update(occ, dt)
            if state == State.CLEAN:
                bits[zid] = 1

            rem_total = (self.fsm_times["waitingTime"] if state == State.WAIT
                         else (self.fsm_times["disinfectionTime"]
                               if state == State.CLEAN
                               else self.fsm_times["standbyTime"]))
            zone_states.append({
                "id": zid,
                "state": State.name(state),
                "timer": round(max(0.0, rem_total - fsm.timer), 1),
                "occupied": occ,
            })

        # Transition detection + logging
        if self.last_bits and len(self.last_bits) == self.num_leds:
            for i in range(self.num_leds):
                if bits[i] != self.last_bits[i]:
                    if bits[i] == 1:
                        msg = MSG_ON_CLEANING
                    else:
                        if any_dark:
                            msg = MSG_OFF_DARKNESS
                        elif occupied.get(i, False):
                            msg = MSG_OFF_PERSON
                        else:
                            msg = MSG_OFF_STANDBY
                    self.log_event(i, msg)

        self.last_bits = tuple(bits)
        return bits, zone_states
