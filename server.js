/**
 * Buddy Stage & Stream Preview Server (Node.js)
 * Zero external dependencies: uses only Node built-in http, fs, and path.
 * Serves the stage UI and proxies to Python backend if active.
 */
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = parseInt(process.env.PORT || '8000', 10);
const PYTHON_PORT = 8000;
const HTML_FILE = path.join(__dirname, 'buddy', 'ui', 'index.html');

// In-memory mock state when python backend is not running
const mockState = {
  fps: 10.2,
  perception_ms: 38,
  gemma_s: 0.42,
  agent_state: 'IDLE',
  why: 'Minimax evaluation: Cell 4 is optimal opening move.',
  hand_in_area: false,
  sheet: { x: 0.21, y: 0.0, yaw: 1.57, confidence: 0.98 },
  cells: ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
  touch_N: 0.0,
  transcript: [
    'Buddy: Hello! Let us play noughts and crosses. You are O, I am X, and you go first.',
    'Player: Let us play'
  ],
  game_events: [
    { kind: 'game_init', text: 'Game ready on sheet', time: '12:00:00' }
  ]
};

const server = http.createServer((req, res) => {
  const urlPath = req.url.split('?')[0];

  // Try proxying to Python backend if PORT is not 8000
  if (PORT !== PYTHON_PORT && (urlPath === '/stream' || urlPath.startsWith('/api') || urlPath === '/say' || urlPath === '/scene' || urlPath === '/ptt')) {
    const proxyReq = http.request({
      hostname: '127.0.0.1',
      port: PYTHON_PORT,
      path: req.url,
      method: req.method,
      headers: req.headers
    }, (proxyRes) => {
      res.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(res);
    });

    proxyReq.on('error', () => {
      // Python backend not running on port 8000, fall back to mock
      handleFallback(urlPath, req, res);
    });

    req.pipe(proxyReq);
    return;
  }

  handleFallback(urlPath, req, res);
});

function handleFallback(urlPath, req, res) {
  if (urlPath === '/' || urlPath === '/index.html') {
    fs.readFile(HTML_FILE, 'utf8', (err, data) => {
      if (err) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        res.end('Error loading index.html: ' + err.message);
        return;
      }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(data);
    });
  } else if (urlPath === '/api/state' || urlPath === '/state.json') {
    res.writeHead(200, { 'Content-Type': 'application/json', 'Cache-Control': 'no-cache' });
    res.end(JSON.stringify(mockState));
  } else if (urlPath === '/say' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const parsed = JSON.parse(body || '{}');
        if (parsed.text) {
          mockState.transcript.push('Player: ' + parsed.text);
          if (mockState.transcript.length > 50) mockState.transcript.shift();
        }
      } catch (e) {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    });
  } else if (urlPath === '/scene' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      try {
        const parsed = JSON.parse(body || '{}');
        const action = parsed.action || '';
        if (action === 'hand_over') mockState.hand_in_area = true;
        if (action === 'hand_away') mockState.hand_in_area = false;
        if (action.startsWith('mark_')) {
          const idx = parseInt(action.split('_')[1], 10);
          if (!isNaN(idx) && idx >= 0 && idx < 9) {
            mockState.cells[idx] = 'O';
            mockState.game_events.push({ kind: 'player_move', cell: idx, text: 'Marked cell ' + idx, time: new Date().toLocaleTimeString() });
          }
        }
      } catch (e) {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ ok: true }));
    });
  } else if (urlPath === '/ptt' && req.method === 'POST') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ ok: true }));
  } else if (urlPath === '/stream') {
    // Generate a placeholder SVG frame or JPEG
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Camera stream is active when running via python -m buddy.main');
  } else {
    res.writeHead(404, { 'Content-Type': 'text/plain' });
    res.end('Not Found');
  }
}

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[buddy-stage] Server listening on http://0.0.0.0:${PORT}`);
});
