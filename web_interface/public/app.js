const socket = new WebSocket('ws://' + window.location.host);

const statusEl = document.getElementById('status');
const neuronGrid = document.getElementById('neuronGrid');
const synapseCanvas = document.getElementById('synapseCanvas');
const ctx = synapseCanvas.getContext('2d');
const tickEl = document.getElementById('tick');
const foodEl = document.getElementById('food');
const dangerEl = document.getElementById('danger');
const targetTypeEl = document.getElementById('targetType');

const agentEl = document.getElementById('agent');
const targetEl = document.getElementById('target');

// Configuration
const GRID_SIZE = 6;
const NEURON_COUNT = 36;
const NEURON_SIZE = 40; // px
const GAP = 60; // px
let initializedCoords = false;

// State
let neurons = []; // { id, el, x, y, v }
let connections = []; // { from, to, active, conf } - we'll update this from server

// Initialize Grid Layout
function initGrid() {
    // CRITICAL FIX: clearing innerHTML removes the canvas!
    // We must re-append it or avoid clearing it.
    neuronGrid.innerHTML = '';
    neuronGrid.appendChild(synapseCanvas); // Re-attach the canvas

    neurons = [];

    for (let i = 0; i < NEURON_COUNT; i++) {
        const el = document.createElement('div');
        el.classList.add('neuron');
        el.textContent = i;


        // Assign types
        if (i === 1 || i === 3) el.classList.add('disabled');
        else if (i < 4) el.classList.add('sensor');
        else if (i === 4) el.classList.add('motor', 'motor-left');
        else if (i === 5) el.classList.add('motor', 'motor-right');

        neuronGrid.appendChild(el);
        neurons.push({ id: i, el, v: 0 });
    }

    // Attempt to calculate positions repeatedly until successful
    const tryUpdate = () => {
        if (!updatePositions()) {
            requestAnimationFrame(tryUpdate);
        } else {
            initializedCoords = true;
        }
    };
    requestAnimationFrame(tryUpdate);
}

const DEBUG_DRAWING = true;

function updatePositions() {
    const gridRect = neuronGrid.getBoundingClientRect();
    if (DEBUG_DRAWING) console.log(`[Pos] Grid Rect: ${gridRect.width}x${gridRect.height}, TopLeft: ${gridRect.top},${gridRect.left}`);

    if (gridRect.width === 0) return false;

    synapseCanvas.width = gridRect.width;
    synapseCanvas.height = gridRect.height;

    if (DEBUG_DRAWING) console.log(`[Pos] Canvas Resized to: ${synapseCanvas.width}x${synapseCanvas.height}`);

    neurons.forEach(n => {
        const rect = n.el.getBoundingClientRect();
        // rect.left is viewport, gridRect.left is viewport.
        // Difference is position within the grid wrapper.
        n.x = rect.left - gridRect.left + rect.width / 2;
        n.y = rect.top - gridRect.top + rect.height / 2;
    });

    if (connections.length > 0) drawSynapses();
    return true;
}

// Draw Connections
function drawSynapses() {
    // if (DEBUG_DRAWING) console.log(`[Draw] Clearing canvas ${synapseCanvas.width}x${synapseCanvas.height}`);
    ctx.clearRect(0, 0, synapseCanvas.width, synapseCanvas.height);

    if (!connections || connections.length === 0) return;

    // if (DEBUG_DRAWING) console.log(`[Draw] Drawing ${connections.length} connections.`);

    ctx.save();
    connections.forEach((conn, index) => {
        const source = neurons[conn.s];
        const target = neurons[conn.t];

        if (!source || !target) {
            if (index === 0 && DEBUG_DRAWING) console.error('[Draw] Invalid neuron refs for conn 0', conn.s, conn.t);
            return;
        }

        // Debug first connection coordinates
        if (index === 0 && DEBUG_DRAWING) {
            console.log(`[Draw] Conn 0: (${source.x.toFixed(1)}, ${source.y.toFixed(1)}) -> (${target.x.toFixed(1)}, ${target.y.toFixed(1)}) Active: ${conn.a}`);
        }

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (conn.b) {
            // Causal Chain: Bright Blue, Thick
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 18;
        } else if (conn.s < 4) {
            // Sensory Outgoing: Yellow
            ctx.strokeStyle = '#facc15';
            ctx.lineWidth = 1.5;
        } else if (conn.t >= 4 && conn.t < 6) {
            // Motor Incoming: Green
            ctx.strokeStyle = '#22c55e';
            ctx.lineWidth = 1.5;
        } else if (conn.a) {
            // Other Active: Red, Opaque
            ctx.strokeStyle = 'rgba(239, 68, 68, 1.0)';
            ctx.lineWidth = 1.5;
        } else {
            // Inactive: White, visible
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.lineWidth = 1;
        }

        ctx.stroke();
    });
    ctx.restore();
}


function updateVisuals(data) {
    // 1. Update Neurons
    if (data.neurons) {
        data.neurons.forEach(nState => {
            const n = neurons[nState.id];
            if (n) {
                if (nState.s) { // Spiked
                    n.el.classList.add('spiked');
                    n.el.style.backgroundColor = ''; // Reset override
                } else {
                    n.el.classList.remove('spiked');
                    const intensity = Math.max(0, Math.min(1, nState.v / 10));
                    n.el.style.backgroundColor = `rgb(${51 + (251 - 51) * intensity}, ${65 + (191 - 65) * intensity}, ${85 + (36 - 85) * intensity})`;
                }
            }
        });
    }

    // 2. Update Connections
    if (data.synapses) {
        if (!initializedCoords) {
            if (updatePositions()) {
                initializedCoords = true;
                console.log('[Visuals] Coords initialized');
            } else {
                console.warn('[Visuals] Failed to init coords (grid width 0)');
            }
        }
        connections = data.synapses;
        drawSynapses();
    } else {
        console.warn('[Visuals] Frame missing synapses data!');
    }

    // 3. Update Stats & World
    if (data.t !== undefined) tickEl.textContent = data.t;

    if (data.world) {
        foodEl.textContent = data.world.food;
        dangerEl.textContent = data.world.danger;

        const typeMap = { 0: 'NONE', 1: 'FOOD', 2: 'DANGER' };
        targetTypeEl.textContent = typeMap[data.world.type] || '?';

        // Update positions on track (Range 0-30)
        // Track width is fixed, calculate percentage
        const trackWidth = document.getElementById('worldTrack').clientWidth;
        const scale = (trackWidth - 20) / 30; // -20 for entity size

        agentEl.style.left = (data.world.agent * scale) + 'px';

        if (data.world.type === 0) {
            targetEl.style.display = 'none';
        } else {
            targetEl.style.display = 'flex';
            targetEl.style.left = (data.world.target * scale) + 'px';

            if (data.world.type === 2) { // Danger
                targetEl.classList.add('danger');
                targetEl.textContent = 'D';
            } else {
                targetEl.classList.remove('danger');
                targetEl.textContent = 'F';
            }
        }
    }
}

// WebSocket Logic
socket.onopen = () => {
    statusEl.textContent = 'Connected';
    statusEl.classList.remove('disconnected');
    statusEl.classList.add('connected');

    // Send initial speed
    const val = document.getElementById('speedRange').value;
    socket.send('speed ' + val);
};

socket.onclose = () => {
    statusEl.textContent = 'Disconnected';
    statusEl.classList.remove('connected');
    statusEl.classList.add('disconnected');
};

socket.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        if (data.log) {
            console.log(data.log);
        } else {
            updateVisuals(data);
        }
    } catch (e) {
        console.error('Parse error:', e);
    }
};

// Controls
document.getElementById('btnStart').addEventListener('click', () => {
    socket.send('start');
});
document.getElementById('btnStop').addEventListener('click', () => {
    socket.send('stop');
});
document.getElementById('btnPause').addEventListener('click', () => {
    socket.send('pause');
});
document.getElementById('btnReset').addEventListener('click', () => {
    socket.send('reset');
});
const speedRange = document.getElementById('speedRange');
const speedInput = document.getElementById('speedInput');
const speedVal = document.getElementById('speedVal');

function updateSpeed(val) {
    speedRange.value = val;
    speedInput.value = val;
    speedVal.textContent = val;
    socket.send('speed ' + val);
}

speedRange.addEventListener('input', (e) => updateSpeed(e.target.value));
speedInput.addEventListener('input', (e) => updateSpeed(e.target.value));

// Handle window resize
window.addEventListener('resize', updatePositions);

// Init
initGrid();
