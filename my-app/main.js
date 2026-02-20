// main.js - robust dev + packaged backend launcher for Electron
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

let mainWindow = null;
let backendProc = null;

const isDev = process.env.ELECTRON_DEV === '1' || process.env.NODE_ENV === 'development';

// Helper: try a list of candidate paths and return first that exists
function findExisting(...candidates) {
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch (e) {}
  }
  return null;
}

// Determine backend command for dev vs packaged
function getBackendSpec() {
  if (isDev) {
    // run the Python script directly in dev
    const python = process.env.PYTHON_PATH || 'python';
    // __dirname here is the directory containing main.js in dev
    const script = path.join(__dirname, 'backend', 'server.py');
    return { cmd: python, args: [script] };
  } else {
    // packaged mode: look for the bundled executable in resources
    // electron-builder's extraResources often ends up in process.resourcesPath
    const resourcesPath = process.resourcesPath || path.join(__dirname, '..', 'resources');
    // try a few plausible locations (plain exe, subdir, .app Contents/MacOS)
    const candidates = [
      path.join(resourcesPath, 'backend'),                       // resources/backend
      path.join(resourcesPath, 'bin', 'backend'),               // resources/bin/backend
      path.join(resourcesPath, 'backend', 'backend'),           // resources/backend/backend
      path.join(resourcesPath, 'backend.app', 'Contents', 'MacOS', 'backend'), // mac .app nested
      // fallback to a build/ folder in project root (useful for local packaging tests)
      path.join(__dirname, 'build', 'backend'),
      path.join(__dirname, 'build', 'backend', 'backend')
    ];
    const exe = findExisting(...candidates);
    return { cmd: exe, args: [] };
  }
}

function spawnBackend() {
  if (backendProc) return backendProc;

  const spec = getBackendSpec();
  if (!spec.cmd) {
    console.error('[main] no backend executable/script found for this mode. Spec candidates were not present.');
    return null;
  }

  console.log('[main] starting backend:', spec.cmd, spec.args.join(' '));

  // spawn the process and capture stdout/stderr
  backendProc = spawn(spec.cmd, spec.args, { stdio: ['ignore', 'pipe', 'pipe'] });

  backendProc.stdout.on('data', (chunk) => {
    const s = chunk.toString();
    // forward to main console
    process.stdout.write('[backend out] ' + s);
    // forward to renderer if available
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('backend-log', { stream: 'stdout', text: s });
    }
  });

  backendProc.stderr.on('data', (chunk) => {
    const s = chunk.toString();
    process.stderr.write('[backend err] ' + s);
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('backend-log', { stream: 'stderr', text: s });
    }
  });

  backendProc.on('exit', (code, signal) => {
    console.log('[backend] exited', code, signal);
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('backend-exit', { code, signal });
    }
    backendProc = null;
  });

  backendProc.on('error', (err) => {
    console.error('[backend] spawn error', err);
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('backend-exit', { error: String(err) });
    }
    backendProc = null;
  });

  return backendProc;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js') // optional: use a preload to receive backend-log via ipcRenderer
    }
  });

  // In dev, load localhost if you use a dev server; otherwise load file
  if (isDev && process.env.ELECTRON_START_URL) {
    mainWindow.loadURL(process.env.ELECTRON_START_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Clean up backend process when quitting
function teardownBackend() {
  if (!backendProc) return;
  try {
    console.log('[main] killing backend process...');
    // first try a graceful kill
    backendProc.kill();
    // if it doesn't exit in a bit, force kill
    setTimeout(() => {
      if (backendProc) {
        try { backendProc.kill('SIGKILL'); } catch (e) {}
      }
    }, 1500);
  } catch (e) {
    console.error('[main] error killing backend:', e);
  }
}

app.on('ready', () => {
  console.log('[main] app ready; isDev=', isDev);
  // start the backend first so it's ready when the renderer connects
  spawnBackend();
  createWindow();
});

app.on('before-quit', () => {
  teardownBackend();
});

app.on('window-all-closed', () => {
  // On macOS apps commonly stay open until explicit quit
  if (process.platform !== 'darwin') {
    teardownBackend();
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) createWindow();
});

// Expose a tiny IPC so renderer can ask main to start/stop backend (optional)
ipcMain.handle('backend-start', async () => {
  return !!spawnBackend();
});
ipcMain.handle('backend-stop', async () => {
  teardownBackend();
  return true;
});
