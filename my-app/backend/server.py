# backend/server.py
import asyncio
import websockets
import json
import os
import sys
import subprocess
import threading
import time
import inspect

WS_HOST = '127.0.0.1'
WS_PORT = 8765

BASE_DIR = os.path.dirname(__file__)
LAUNCHER_PY = os.path.join(BASE_DIR, 'suvos_launcher.py')

clients = set()
q = asyncio.Queue()

def push(item):
    """
    Thread-safe push into the asyncio queue used by the websocket handler.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(q.put(item), loop)
    except RuntimeError:
        # no running loop in this thread; ignore the push
        pass

# ---------- WebSocket handler (robust to either (ws) or (ws, path)) ----------
async def ws_handler(ws, path=None):
    """
    Handle a single websocket connection.
    Accepts either signature: (ws) or (ws, path). If path is None, try to
    extract it from the ws object (different websockets versions put it in
    different attributes).
    """
    # determine path robustly
    if path is None:
        # try common places where the path may live
        path = getattr(ws, "path", None)
        if not path:
            # some versions expose a 'request' object with path
            req = getattr(ws, "request", None)
            if req is not None:
                path = getattr(req, "path", None)
            # other versions may have request.uri or request.path
            if not path:
                path = getattr(ws, "remote_address", None)
    try:
        print(f"Client connected: path={path}")
        clients.add(ws)

        # send init so renderer knows config/room shape
        init = {
            "type": "INIT_CONFIG",
            "config": {"scale": 20, "off_x": 0, "off_y": 0, "map_size": 800},
        }
        try:
            # attempt to find room_shape relative to project structure
            room_path = os.path.join(os.path.dirname(__file__), '..', 'renderer', 'room_shape.json')
            if os.path.exists(room_path):
                with open(room_path, 'r') as f:
                    init['room_shape'] = json.load(f)
            else:
                init['room_shape'] = None
        except Exception:
            init['room_shape'] = None

        # send init (wrap in try in case client disconnects immediately)
        try:
            await ws.send(json.dumps(init))
        except Exception as e:
            print("Initial send failed:", e)
            return

        # keep sending queued messages to this client
        while True:
            msg = await q.get()
            if msg is None:
                break
            if isinstance(msg, (dict, list)):
                payload = json.dumps(msg)
            else:
                payload = str(msg)
            try:
                await ws.send(payload)
            except Exception as e:
                print("ws send failed, closing connection:", e)
                break
    except Exception as e:
        print("ws_handler exception:", repr(e))
    finally:
        clients.discard(ws)
        print("Client disconnected")


# Diagnostic: show what function signature is bound at runtime
try:
    sig = inspect.signature(ws_handler)
    print(f"[DIAG] ws_handler signature: {sig}  (defined in {ws_handler.__code__.co_filename})")
except Exception as e:
    print("[DIAG] cannot inspect ws_handler:", e)


# ---------- launcher / subprocess helpers ----------
def try_import_launcher():
    """
    Try to import and run the launcher directly only if it is known safe.
    Otherwise spawn it as a subprocess to avoid importing GUI toolkits like tkinter
    into this process. This function prefers spawning to avoid GUI libs in server.
    """
    launcher_py = os.path.join(BASE_DIR, 'suvos_launcher.py')
    if os.path.exists(launcher_py):
        try:
            proc = subprocess.Popen([sys.executable, launcher_py],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True)
            # forward stdout/stderr to websocket queue
            def read_stdout():
                for line in proc.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        push(json.loads(line))
                    except Exception:
                        push({"type": "LOG", "payload": line})
            def read_stderr():
                for line in proc.stderr:
                    line = line.strip()
                    if line:
                        push({"type":"BACKEND_ERR","payload":line})
            threading.Thread(target=read_stdout, daemon=True).start()
            threading.Thread(target=read_stderr, daemon=True).start()
            return proc
        except Exception as e:
            push({"type":"BACKEND_ERR","payload":f"Failed to spawn launcher: {e}"})
            return None
    return None


def spawn_launcher_process():
    """
    Spawn the launcher as a subprocess and forward its stdout/stderr into the queue.
    """
    if os.path.exists(LAUNCHER_PY):
        proc = subprocess.Popen([sys.executable, LAUNCHER_PY], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def read_stdout():
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    push(parsed)
                except Exception:
                    push({"type":"LOG", "payload": line})
        def read_stderr():
            for l in proc.stderr:
                l = l.strip()
                if l:
                    push({"type":"BACKEND_ERR", "payload": l})
        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()
        return proc
    return None


# ---------- fallback emulator for local testing ----------
def emulator_loop():
    i = 0
    while True:
        i += 1
        push({
            "type": "TEST_UPDATE",
            "timestamp": time.time(),
            "people": [{"id": 1, "x": 100 + (i % 200), "y": 200 + ((i * 3) % 200), "confidence": 0.9}]
        })
        time.sleep(1)


# ---------- main server entrypoint ----------
async def main():
    print(f"Starting WS at ws://{WS_HOST}:{WS_PORT}")
    # register the ws_handler (signature is ws, path=None which works for both styles)
    server = await websockets.serve(ws_handler, WS_HOST, WS_PORT)
    print("[DIAG] websockets.serve registered ws_handler")

    # try to start the launcher (prefer not to import GUI libs into this process)
    started = try_import_launcher()
    proc = None
    if not started:
        proc = spawn_launcher_process()

    if not started and not proc:
        # no launcher found; start emulator for UI testing
        threading.Thread(target=emulator_loop, daemon=True).start()

    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down")
