import init, { WasmRuntime } from '../pkg/engram_wasm.js';

const GRID_SIZE = 12;
const CELL_SIZE = 40;
const MODULE_NAMES = ['Sensory', 'Assoc.', 'Predict', 'Episodic', 'Action', 'Safety'];
const MODULE_COLORS = ['#00d4ff', '#a855f7', '#ff8c00', '#22d3ee', '#22c55e', '#ef4444'];
const MODULE_NEURON_COUNTS = [128, 256, 64, 64, 128, 32];
const MODULE_OFFSETS = [0, 128, 384, 448, 512, 640];
const ACTION_ARROWS = ['\u2191', '\u2192', '\u2193', '\u2190']; // up right down left

let runtime: WasmRuntime | null = null;
let currentTool = 1;
let paused = false;
let speed = 10;
let spikeHistory: number[][] = [];

const gridCanvas = document.getElementById('grid-canvas') as HTMLCanvasElement;
const gridCtx = gridCanvas.getContext('2d')!;
const spikeCanvas = document.getElementById('spike-canvas') as HTMLCanvasElement;
const spikeCtx = spikeCanvas.getContext('2d')!;

// Build module bars
const barsContainer = document.getElementById('module-bars')!;
MODULE_NAMES.forEach((name, i) => {
  const bar = document.createElement('div');
  bar.className = 'module-bar';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'module-name';
  nameSpan.style.color = MODULE_COLORS[i];
  nameSpan.textContent = name;

  const track = document.createElement('div');
  track.className = 'bar-track';
  const fill = document.createElement('div');
  fill.className = 'bar-fill';
  fill.id = `bar-${i}`;
  fill.style.background = MODULE_COLORS[i];
  fill.style.width = '0%';
  track.appendChild(fill);

  bar.appendChild(nameSpan);
  bar.appendChild(track);
  barsContainer.appendChild(bar);
});

// Tool buttons
document.querySelectorAll('[data-tool]').forEach(btn => {
  btn.addEventListener('click', () => {
    currentTool = parseInt((btn as HTMLElement).dataset.tool || '1');
    document.querySelectorAll('[data-tool]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// Reset button
document.getElementById('reset-btn')!.addEventListener('click', () => {
  if (runtime) {
    runtime.reset();
    spikeHistory = [];
  }
});

// Pause button
document.getElementById('pause-btn')!.addEventListener('click', () => {
  paused = !paused;
  document.getElementById('pause-btn')!.textContent = paused ? 'Play' : 'Pause';
});

// Speed slider
const speedSlider = document.getElementById('speed-slider') as HTMLInputElement;
speedSlider.addEventListener('input', () => {
  speed = parseInt(speedSlider.value);
  document.getElementById('speed-label')!.textContent = speed + 'x';
});

// Grid click
gridCanvas.addEventListener('click', (e: MouseEvent) => {
  if (!runtime) return;
  const rect = gridCanvas.getBoundingClientRect();
  const x = Math.floor((e.clientX - rect.left) / CELL_SIZE);
  const y = Math.floor((e.clientY - rect.top) / CELL_SIZE);
  if (x >= 0 && x < GRID_SIZE && y >= 0 && y < GRID_SIZE) {
    runtime.set_cell(x, y, currentTool);
  }
});

function drawGrid() {
  if (!runtime) return;

  const grid = runtime.grid_data();
  const ax = runtime.agent_x();
  const ay = runtime.agent_y();
  const tx = runtime.target_x();
  const ty = runtime.target_y();

  gridCtx.fillStyle = '#0a0a0f';
  gridCtx.fillRect(0, 0, gridCanvas.width, gridCanvas.height);

  for (let y = 0; y < GRID_SIZE; y++) {
    for (let x = 0; x < GRID_SIZE; x++) {
      const cell = grid[y * GRID_SIZE + x];
      const px = x * CELL_SIZE;
      const py = y * CELL_SIZE;

      // Cell background
      const colors: Record<number, string> = { 1: '#2a2a4e', 2: '#22c55e22', 3: '#ef444422' };
      gridCtx.fillStyle = colors[cell] || '#12121a';
      gridCtx.fillRect(px + 1, py + 1, CELL_SIZE - 2, CELL_SIZE - 2);

      // Cell content
      gridCtx.font = '20px serif';
      gridCtx.textAlign = 'center';
      gridCtx.textBaseline = 'middle';
      if (cell === 1) { gridCtx.fillStyle = '#666'; gridCtx.fillText('#', px + CELL_SIZE / 2, py + CELL_SIZE / 2); }
      if (cell === 3) { gridCtx.fillStyle = '#ef4444'; gridCtx.fillText('!', px + CELL_SIZE / 2, py + CELL_SIZE / 2); }
      if (x === tx && y === ty) { gridCtx.fillStyle = '#22c55e'; gridCtx.fillText('$', px + CELL_SIZE / 2, py + CELL_SIZE / 2); }
    }
  }

  // Agent
  gridCtx.fillStyle = '#00ffaa';
  gridCtx.beginPath();
  gridCtx.arc(ax * CELL_SIZE + CELL_SIZE / 2, ay * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE / 3, 0, Math.PI * 2);
  gridCtx.fill();
  gridCtx.fillStyle = '#0a0a0f';
  gridCtx.font = 'bold 16px monospace';
  gridCtx.textAlign = 'center';
  gridCtx.textBaseline = 'middle';
  gridCtx.fillText('A', ax * CELL_SIZE + CELL_SIZE / 2, ay * CELL_SIZE + CELL_SIZE / 2 + 1);

  // Grid lines
  gridCtx.strokeStyle = 'rgba(255,255,255,0.03)';
  gridCtx.lineWidth = 0.5;
  for (let i = 0; i <= GRID_SIZE; i++) {
    gridCtx.beginPath(); gridCtx.moveTo(i * CELL_SIZE, 0); gridCtx.lineTo(i * CELL_SIZE, GRID_SIZE * CELL_SIZE); gridCtx.stroke();
    gridCtx.beginPath(); gridCtx.moveTo(0, i * CELL_SIZE); gridCtx.lineTo(GRID_SIZE * CELL_SIZE, i * CELL_SIZE); gridCtx.stroke();
  }
}

function drawSpikes() {
  const w = spikeCanvas.width;
  const h = spikeCanvas.height;
  spikeCtx.fillStyle = '#0a0a0f';
  spikeCtx.fillRect(0, 0, w, h);

  const maxCols = 200;
  const colW = w / maxCols;
  const totalNeurons = 672;
  const rowH = h / totalNeurons;

  for (let col = 0; col < spikeHistory.length; col++) {
    const activities = spikeHistory[col];
    if (!activities) continue;
    for (let m = 0; m < 6; m++) {
      const level = activities[m] || 0;
      const count = Math.floor(level * MODULE_NEURON_COUNTS[m] * 0.3);
      for (let s = 0; s < count; s++) {
        const nid = MODULE_OFFSETS[m] + Math.floor(Math.random() * MODULE_NEURON_COUNTS[m]);
        spikeCtx.fillStyle = MODULE_COLORS[m];
        spikeCtx.globalAlpha = 0.7;
        spikeCtx.fillRect(col * colW, nid * rowH, Math.max(colW, 1), Math.max(rowH, 1));
      }
    }
  }
  spikeCtx.globalAlpha = 1;
}

function updateMetrics() {
  if (!runtime) return;
  document.getElementById('m-tick')!.textContent = String(runtime.get_tick());
  const reward = runtime.get_total_reward();
  const rewardEl = document.getElementById('m-reward')!;
  rewardEl.textContent = reward.toFixed(2);
  rewardEl.style.color = reward >= 0 ? '#22c55e' : '#ef4444';
  document.getElementById('m-error')!.textContent = runtime.get_prediction_error().toFixed(3);
  document.getElementById('m-action')!.textContent = ACTION_ARROWS[runtime.get_current_action()] || '-';
}

function updateModuleBars() {
  if (!runtime) return;
  const activities = runtime.module_activities();
  for (let i = 0; i < 6; i++) {
    const pct = Math.min(100, (activities[i] || 0) * 100);
    document.getElementById(`bar-${i}`)!.style.width = pct + '%';
  }
}

function gameLoop() {
  if (!paused && runtime) {
    for (let i = 0; i < speed; i++) {
      runtime.step();
    }

    const activities = Array.from(runtime.module_activities());
    spikeHistory.push(activities);
    if (spikeHistory.length > 200) spikeHistory.shift();

    // Auto-reset on target reached
    if (runtime.agent_x() === runtime.target_x() && runtime.agent_y() === runtime.target_y()) {
      runtime.reset();
    }
  }

  drawGrid();
  drawSpikes();
  updateMetrics();
  updateModuleBars();
  requestAnimationFrame(gameLoop);
}

async function start() {
  try {
    await init();
    runtime = new WasmRuntime(GRID_SIZE);

    // Border walls
    for (let i = 0; i < GRID_SIZE; i++) {
      runtime.set_cell(i, 0, 1);
      runtime.set_cell(i, GRID_SIZE - 1, 1);
      runtime.set_cell(0, i, 1);
      runtime.set_cell(GRID_SIZE - 1, i, 1);
    }
    // Interior walls
    runtime.set_cell(4, 2, 1); runtime.set_cell(4, 3, 1); runtime.set_cell(4, 4, 1);
    runtime.set_cell(7, 5, 1); runtime.set_cell(7, 6, 1); runtime.set_cell(7, 7, 1);
    runtime.set_cell(3, 8, 1); runtime.set_cell(4, 8, 1);
    // Hazards
    runtime.set_cell(5, 5, 3); runtime.set_cell(8, 3, 3); runtime.set_cell(3, 9, 3);

    const statusEl = document.getElementById('status')!;
    statusEl.textContent = 'Running';
    statusEl.style.color = '#22c55e';

    gameLoop();
  } catch (err) {
    const statusEl = document.getElementById('status')!;
    statusEl.textContent = 'Error: ' + (err as Error).message;
    statusEl.style.color = '#ef4444';
    console.error(err);
  }
}

start();
