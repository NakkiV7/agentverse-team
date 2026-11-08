// Penny's demo page. Plain JS, no build step, nothing fetched from outside: the demo may run offline.
// It polls state.json and the three images twice a second and not at all while the tab is hidden (the app renders
// an image only while the page fetches it, and every render competes with the language model for the GPU).
"use strict";

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const post = (url, body) => fetch(url, { method: "POST", body: JSON.stringify(body) });

// ---------------------------------------------------------------- words for the player
const STAGES = {                          // legend: shown as a tooltip on every badge, and on the filter buttons
  hear: "HEAR: microphone, Silero VAD and Whisper, or the keyboard", understand: "UNDERSTAND: Gemma, structured output",
  decide: "DECIDE: the state machine", see: "SEE: overhead camera, OpenCV and the mark CNN",
  plan: "PLAN: reach checks, the minimax solver", act: "ACT: the arm controller", sense: "SENSE: pen-tip touch sensor",
  check: "CHECK: the camera verifies the ink", adapt: "ADAPT: re-plan when the scene changes",
  safety: "SAFETY: the depth-camera reflex, the arm freezes", say: "SAY: the Kokoro voice",
  system: "SYSTEM: start-up and scene changes" };
const GROUP = { see: "see", check: "see", sense: "see", safety: "see", plan: "plan", act: "plan", adapt: "plan",
                hear: "decide", understand: "decide", decide: "decide", say: "say", system: "system" };
const NAMES = ["top left", "top middle", "top right", "middle left", "centre", "middle right", "bottom left", "bottom middle", "bottom right"];
const WORDS = {                           // the state machine's name -> what a person would say is happening
  IDLE: "Resting", CHAT: "Chatting", CHOOSE_GAME: "Choosing a game", NAME_GAME: "Naming your game",
  TEACH: "Learning your game", TEACH_MODEL: "Learning your game", OFFER_TAUGHT: "Ready to play your game",
  SETUP: "Penny is drawing the board", DOTS_SETUP: "Penny is drawing the dots", HANGMAN_SETUP: "Penny is setting up hangman",
  WAIT_SHEET: "Penny needs the paper", ASK_LEVEL: "Choosing a level", ASK_SIDE: "Choosing sides", DOTS_SIZE: "Choosing the board size",
  PLAYER_TURN: "Your turn", DOTS_PLAYER_TURN: "Your turn", HANGMAN_TURN: "Your guess",
  ROBOT_TURN: "Penny is drawing", DOTS_ROBOT_TURN: "Penny is drawing", CONFIRM_MARK: "Penny has a question", GAME_OVER: "Game over" };
const NEXT = {                            // the one line above the input: what to do now
  IDLE: "Say or type <b>Penny</b> to wake it up.",
  CHAT: "Ask for a game: <b>let's play a game</b>. Or teach it one: <b>let me teach you a game</b>.",
  CHOOSE_GAME: "Name a game Penny knows (hover the game name at the top to see them), or say <b>let me teach you a game</b>.",
  NAME_GAME: "Tell Penny what to call the game you just taught it.",
  TEACH: "Explain your game in your own words, a sentence or two at a time. Penny asks about anything it is unsure of.",
  TEACH_MODEL: "Explain your game in your own words, a sentence or two at a time. Penny asks about anything it is unsure of.",
  OFFER_TAUGHT: "Say <b>yes</b> to play the game you taught, or <b>no</b>.",
  SETUP: "Penny is drawing the board. Watch the arm, and its reasoning under How Penny decided.",
  WAIT_SHEET: "Penny needs the paper back: press <b>Put it back</b> or <b>New blank sheet</b> under Test it.",
  ASK_LEVEL: "Choose how hard Penny plays: <b>easy</b>, <b>medium</b>, <b>hard</b>, <b>impossible</b>, or <b>authentic</b> (it learns your level over a few games).",
  ASK_SIDE: "Say <b>crosses</b> or <b>noughts</b>. Crosses go first.",
  PLAYER_TURN: "Your move: draw your mark on the paper, press a square, or say it: <b>put mine in the top left</b>.",
  CONFIRM_MARK: "Penny asked about a mark it saw. Answer <b>yes</b> or <b>no</b>.",
  ROBOT_TURN: "Penny's move: it plans, draws, then reads its own mark back. Try <b>Reach over the paper</b> while it draws.",
  DOTS_SIZE: "Say the board size: <b>three by three</b>, <b>four across and three down</b>, or <b>the usual</b>.",
  DOTS_SETUP: "Penny is drawing the dots. Watch the arm.",
  DOTS_PLAYER_TURN: "Your line: draw it on the paper between two neighbouring dots, or say it: <b>the top side of the top left box</b>. Close a box and you go again.",
  DOTS_ROBOT_TURN: "Penny's line: it plans, draws, then reads its own line back.",
  HANGMAN_SETUP: "Penny is writing the word's blanks on the paper. Watch the arm.",
  HANGMAN_TURN: "Guess a letter: say or type it, for instance <b>E</b>. You can also ask for a hint, or ask what you have tried.",
  GAME_OVER: "Say <b>yes</b> for another game (press <b>New blank sheet</b> first), or ask Penny anything about the one you played.",
};
const GRID_STATES = new Set(["SETUP", "ASK_SIDE", "PLAYER_TURN", "ROBOT_TURN", "CONFIRM_MARK"]);   // the nine squares show
const color = s => getComputedStyle(document.documentElement).getPropertyValue("--" + s).trim() || "#888";

let S = null, micOn = true, nTrace = -1, nChat = -1, partial = "", hintTimer = null;
const myTurn = () => !!S && (S.state === "PLAYER_TURN" || S.state === "DOTS_PLAYER_TURN" || S.example);

function warn(text) {                     // a short warning under the paper, gone after four seconds
  const el = $("padline"); el.className = "hint warn"; el.textContent = text;
  clearTimeout(hintTimer); hintTimer = setTimeout(() => { hintTimer = null; padline(); }, 4000);
}
function notNow() {
  warn(!S ? "" : S.state === "ROBOT_TURN" || S.state === "DOTS_ROBOT_TURN" ? "It's Penny's turn: wait for it to finish." :
    S.state === "CONFIRM_MARK" ? "Answer Penny's question first (yes or no)." :
    S.state === "HANGMAN_TURN" ? "In hangman you say or type a letter; Penny writes it." :
    "There's no game on yet: say Penny, then ask for a game. Penny draws the board before you mark it.");
}
function padline() {                      // the line under the paper: a drawing hint only while drawing is possible
  if (hintTimer) return;
  const el = $("padline");
  if (myTurn()) { el.className = "hint draw"; el.textContent = S.example ? "Draw a short line anywhere on the paper above, with the mouse." :
    "Draw on the paper above with the mouse: your hand in the simulation draws it on the real sheet."; }
  else { el.className = "hint"; el.textContent = ""; }
  $("paperframe").classList.toggle("candraw", myTurn());
}

// ---------------------------------------------------------------- the nine squares (grid games only)
$("cells").innerHTML = NAMES.map((n, i) => `<button type="button" data-cell="${i}">${n}<span class="m"></span></button>`).join("");
document.querySelectorAll("[data-cell]").forEach(b => b.addEventListener("click", () => {
  if (!myTurn()) return notNow();
  if (S.state === "DOTS_PLAYER_TURN") return warn("In dots and boxes, draw your line on the paper above.");
  const legal = (S.game && S.game.legal) || [];
  if (legal.length && !legal.includes(+b.dataset.cell)) return warn("In this game your mark has to go next to the last one: use a green square.");
  post("scene", { action: "player_mark", arg: +b.dataset.cell });
}));

// ---------------------------------------------------------------- images: fixed boxes, so a new frame never moves the page
const inflight = {};
function refreshImg(id, url) {
  if (inflight[id]) return;               // never more than one fetch of an image at a time
  inflight[id] = true;
  const im = new Image();
  im.onload = () => { inflight[id] = false; const el = $(id); el.src = im.src; el.parentElement.classList.add("live"); };
  im.onerror = () => { inflight[id] = false; };
  im.src = url + "?t=" + Date.now();
}
setInterval(() => { if (document.hidden) return; refreshImg("overhead", "overhead.jpg"); refreshImg("scene", "scene.jpg"); refreshImg("sheetimg", "sheet.jpg"); }, 500);

// ---------------------------------------------------------------- the drawing pad over the straightened sheet
// The sheet image is 891 x 630 px for 297 x 210 mm (A4 landscape, 3 px per mm); strokes go to the app in sheet
// millimetres with the origin at the centre, x across the sheet, y up.
const pad = $("pad"), ctx = pad.getContext("2d");
let strokes = [], cur = null, sendTimer = null;
function toMM(e) {
  const r = pad.getBoundingClientRect();
  const u = (e.clientX - r.left) / r.width, v = (e.clientY - r.top) / r.height;
  return [u * 297 - 148.5, 105 - v * 210, e.clientX - r.left, e.clientY - r.top];
}
function resizePad() { const r = pad.getBoundingClientRect(); pad.width = r.width; pad.height = r.height; }
new ResizeObserver(resizePad).observe(pad);
pad.addEventListener("pointerdown", e => { if (!myTurn()) return notNow(); clearTimeout(sendTimer); cur = [toMM(e)]; pad.setPointerCapture(e.pointerId); });
pad.addEventListener("pointermove", e => {
  if (!cur) return; const p = toMM(e), q = cur[cur.length - 1];
  if (Math.hypot(p[2] - q[2], p[3] - q[3]) < 3) return;
  ctx.strokeStyle = "rgba(220,40,40,.8)"; ctx.lineWidth = 3; ctx.lineCap = "round"; ctx.beginPath(); ctx.moveTo(q[2], q[3]); ctx.lineTo(p[2], p[3]); ctx.stroke();
  cur.push(p);
});
pad.addEventListener("pointerup", () => {
  if (cur && cur.length > 1) strokes.push(cur.map(p => [p[0], p[1]]));
  cur = null;
  sendTimer = setTimeout(() => {          // send the whole mark (e.g. both strokes of an X) together
    if (strokes.length) post("draw", { strokes }); strokes = [];
    setTimeout(() => ctx.clearRect(0, 0, pad.width, pad.height), 600);
  }, 900);
});

// ---------------------------------------------------------------- header
function setDot(id, cls, text) { const d = $(id); d.className = "dot " + cls; d.title = text; d.setAttribute("aria-label", text); }
function header(s) {
  const g = s.game || {};
  $("gamename").textContent = g.name || "";
  $("gamename").title = g.known ? "Games Penny knows: " + g.known.join(", ") : "";
  const st = $("statetext");
  let words = WORDS[s.state] || s.state, cls = "state";
  if ((s.state === "ROBOT_TURN" || s.state === "DOTS_ROBOT_TURN") && s.arm === "home") words = "Penny is thinking";
  if (s.example) words = "Show Penny an example move";
  if (s.state === "IDLE") words = "Resting: say Penny";
  if (s.state === "PLAYER_TURN" || s.state === "DOTS_PLAYER_TURN" || s.state === "HANGMAN_TURN" || s.example) cls += " yours";
  if (s.safety.frozen) { words = "Arm frozen: hand over the paper"; cls = "state alarm"; }
  st.textContent = words; st.className = cls; st.title = "state machine: " + s.state;
  const lv = $("level");
  lv.textContent = "Level " + (g.level || "adaptive");
  lv.title = `Your rating ${g.rating ?? "-"} after ${g.rating_games ?? 0} game${g.rating_games === 1 ? "" : "s"}` +
    (g.difficulty_reason ? ` · ${g.difficulty_reason}` : "") + ` · Penny plays at ${Math.round(100 * (s.strength || 0))}% of its best`;
  setDot("dot-voice", s.speaking ? "on" : "", s.speaking ? "Voice: Penny is speaking" : "Voice: idle");
  if (!s.mic.available) setDot("dot-mic", "off", "Microphone: none, type below");
  else setDot("dot-mic", s.mic.enabled ? "ok" : "warn", s.mic.enabled ? "Microphone: on, listening" : "Microphone: muted");
  setDot("dot-safety", s.safety.frozen ? "bad" : s.safety.pixels ? "warn" : "ok",
    s.safety.frozen ? `Safety: something over the paper (${s.safety.height_cm} cm high), arm frozen` :
    s.safety.pixels ? "Safety: something near the paper" : "Safety: clear, nothing over the paper");
}

// ---------------------------------------------------------------- what Penny is hearing (another process fills hearing = {active, partial, level})
function hearing(s, newPlayerLine) {
  const h = s.hearing, strip = $("hearing"), input = $("text");
  const available = s.mic.available || !!h;
  strip.classList.toggle("gone", !available);
  $("micbtn").classList.toggle("gone", !s.mic.available);
  if (!available) { $("ghost").textContent = ""; return; }
  // hearing.level is 0..1; mic.level is a raw RMS where 0.07 is loud
  const level = h ? Math.min(100, Math.round(100 * (h.level || 0))) : Math.min(100, Math.round(1500 * (s.mic.level || 0)));
  $("lvl").style.width = (s.mic.available && !s.mic.enabled ? 0 : level) + "%";
  const active = h ? !!h.active : s.mic.enabled;
  if (newPlayerLine || !active) partial = "";          // the final text arrived, or Penny stopped listening
  else if (h && h.partial) partial = String(h.partial);
  const typing = input.value.length > 0;
  $("ghost").textContent = !typing && partial ? partial : "";
  $("ghost").parentElement.classList.toggle("hearing-ghost", !typing && !!partial);
  $("partial").textContent = !active ? (s.mic.available && !s.mic.enabled ? "Microphone muted" : "") :
    typing && partial ? "Hearing: " + partial : partial ? "Hearing you…" : "Listening…";
}
$("text").addEventListener("input", () => { if (S) hearing(S, false); });

// ---------------------------------------------------------------- the trace
function trace(s) {
  const key = s.trace.length ? s.trace[s.trace.length - 1].t + s.trace.length : 0;
  if (key === nTrace) return;
  nTrace = key;
  const el = $("trace"), atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40 || !el.innerHTML;
  el.innerHTML = s.trace.map(r => `<div class="row" data-g="${GROUP[r.stage] || "system"}"><span class="t">${r.t.toFixed(1)}s</span>` +
    `<span class="badge" style="background:${color(r.stage)}" title="${esc(STAGES[r.stage] || r.stage)}">${esc(r.stage.toUpperCase())}</span>` +
    `<span>${esc(r.msg)}</span></div>`).join("");
  if (atBottom) el.scrollTop = el.scrollHeight;
  $("tracecount").textContent = s.trace.length ? `${s.trace.length} steps` : "";
  const last = s.trace[s.trace.length - 1];
  $("tracelast").textContent = last ? `${last.stage.toUpperCase()}: ${last.msg}` : "";
}
document.querySelectorAll("#filters button").forEach(b => b.addEventListener("click", () => {
  document.querySelectorAll("#filters button").forEach(x => x.classList.toggle("on", x === b));
  $("trace").className = b.dataset.f === "all" ? "" : "f-" + b.dataset.f;
}));
if (matchMedia("(min-width: 1180px)").matches) $("think").open = true;    // open on a wide screen, collapsed on a narrow one

// ---------------------------------------------------------------- one poll
async function tick() {
  let s;
  try { s = await (await fetch("state.json", { cache: "no-store" })).json(); } catch (e) { return; }
  S = s;
  header(s);

  // the conversation
  let newPlayerLine = false;
  if (s.transcript.length !== nChat) {
    const last = s.transcript[s.transcript.length - 1];
    newPlayerLine = nChat >= 0 && s.transcript.length > nChat && last && last.who === "Player";
    nChat = s.transcript.length;
    const el = $("chat");
    let lastBuddy = -1; s.transcript.forEach((m, i) => { if (m.who === "Penny") lastBuddy = i; });
    el.innerHTML = s.transcript.length ? s.transcript.map((m, i) =>
      `<div class="msg ${esc(m.who)}${i === lastBuddy ? " latest" : ""}"><small>${m.who === "Player" ? "You" : "Penny"} · ${m.t.toFixed(1)}s` +
      `${i === lastBuddy ? ' <span class="speaking" id="speakmark"></span>' : ""}</small>${esc(m.text)}</div>`).join("") :
      `<p class="empty">Nothing said yet. Penny wakes up when you say its name.</p>`;
    el.scrollTop = el.scrollHeight;
  }
  const sm = $("speakmark"); if (sm) sm.textContent = s.speaking ? "· speaking" : "";
  const next = $("next");
  $("nexthint").innerHTML = s.safety.frozen ? "Penny's arm is frozen while something is over the paper. Move it away and Penny carries on." :
    s.example ? "Show Penny one example move: draw a short line anywhere on the paper, or say <b>skip</b>." : (NEXT[s.state] || "");
  next.classList.toggle("warn", !!s.safety.frozen);
  hearing(s, newPlayerLine);

  // the paper: squares only for a grid game, the drawing hint only when drawing is possible
  const legal = (s.game && s.game.legal) || [], turn = s.state === "PLAYER_TURN";
  const grid = GRID_STATES.has(s.state) && !(s.game && s.game.name === "dots and boxes");
  $("cells").hidden = !grid;
  if (grid) document.querySelectorAll("[data-cell]").forEach(b => {
    const c = +b.dataset.cell, taken = s.board && s.board[c];
    b.classList.toggle("off", !turn || !!taken || (legal.length > 0 && !legal.includes(c)));
    b.classList.toggle("legal", turn && legal.length > 0 && legal.includes(c) && !taken);
    b.querySelector(".m").textContent = taken ? (taken === "X" ? "✕" : "○") : "";
  });
  const sides = s.player && s.robot ? ` You are ${s.player === "X" ? "crosses" : "noughts"}, Penny ${s.robot === "X" ? "crosses" : "noughts"}.` : "";
  $("papercap").textContent = s.state.startsWith("DOTS") ? "The paper, straightened, with the dots and lines Penny has read." :
    s.state.startsWith("HANGMAN") ? "The paper, straightened: the word's blanks, the letters and the gallows." :
    "The paper, straightened, with Penny's reading of each square written on it." + sides;
  padline();

  // the trace and the sensors that used to crowd the header
  trace(s);
  const sh = s.sheet;
  $("sensors").innerHTML = `arm <b>${esc(s.arm)}</b> · pen <b>${s.touch_N.toFixed(2)} N</b> · sheet <b>${sh ? `${sh.x_cm}, ${sh.y_cm} cm · ${sh.rot_deg}° · ${sh.visible ? (sh.stable ? "steady" : "moving") : "hidden"}` : "not seen"}</b>` +
    ` · strength <b>${Math.round(100 * (s.strength || 0))}%</b> · rating <b>${s.game ? s.game.rating : "-"}</b> · sim <b>${s.sim_time}s</b>`;
  micOn = !!s.mic.enabled;
  $("micbtn").textContent = micOn ? "Mute mic" : "Unmute mic";
}
setInterval(() => { if (!document.hidden) tick(); }, 500); tick();
document.addEventListener("visibilitychange", () => { if (!document.hidden) tick(); });

// ---------------------------------------------------------------- the player's actions
$("say").addEventListener("submit", async e => {
  e.preventDefault();
  const t = $("text"); if (!t.value.trim()) return;
  const text = t.value; t.value = ""; partial = ""; $("ghost").textContent = "";
  await post("say", { text });
  t.focus();
});
document.querySelectorAll("[data-a]").forEach(b => b.addEventListener("click", () => post("scene", { action: b.dataset.a })));
$("micbtn").addEventListener("click", () => post("mic", { enabled: !micOn }));

// theme: follows the system until the button is pressed (auto -> light -> dark -> auto), remembered in this browser
const THEMES = ["", "light", "dark"], root = document.documentElement;
let theme = ""; try { theme = localStorage.getItem("buddy-theme") || ""; } catch (e) { /* private window */ }
const q = new URLSearchParams(location.search).get("theme"); if (THEMES.includes(q)) theme = q;     // ?theme=light for a projector
function applyTheme() {
  if (theme) root.setAttribute("data-theme", theme); else root.removeAttribute("data-theme");
  $("theme").textContent = theme === "light" ? "☀" : theme === "dark" ? "☾" : "◐";
  $("theme").title = "Theme: " + (theme || "follows the system") + ". Click to change.";
}
applyTheme();
$("theme").addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try { localStorage.setItem("buddy-theme", theme); } catch (e) { /* fine without */ }
  applyTheme();
});
