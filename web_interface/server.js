const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const LOG_FILE = path.join(__dirname, 'server.log');

function log(message) {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}\n`;
    console.log(message);
    fs.appendFileSync(LOG_FILE, logMessage);
}

// Clear log on startup
fs.writeFileSync(LOG_FILE, `--- Server started at ${new Date().toISOString()} ---\n`);

// Serve static files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Path to C++ executable
const EXE_PATH = path.join(__dirname, '../x64/Debug/binary_rstdp.exe');
const CMD = EXE_PATH;

log(`Targeting executable: ${CMD}`);

wss.on('connection', (ws) => {
    log('Client connected (WebSocket)');
    log('Spawning C++ process...');

    let buffer = '';
    const process = spawn(CMD, [], {
        cwd: path.join(__dirname, '../')
    });

    let lastFrameTime = 0;
    const FRAME_INTERVAL = 33; // ~30 FPS

    process.stdout.on('data', (data) => {
        const rawData = data.toString();
        buffer += rawData;

        let boundary = buffer.indexOf('\n');
        while (boundary !== -1) {
            const line = buffer.substring(0, boundary).trim();
            buffer = buffer.substring(boundary + 1);

            if (line.startsWith('{') && line.endsWith('}')) {
                const now = Date.now();
                if (now - lastFrameTime >= FRAME_INTERVAL) {
                    if (ws.readyState === WebSocket.OPEN) {
                        // Check backpressure
                        if (ws.bufferedAmount < 1024 * 1024) { // 1MB limit
                            ws.send(line);
                            lastFrameTime = now;
                        } else {
                            // Drop frame if buffer is too full
                        }
                    }
                }
            } else if (line.length > 0) {
                log(`[CPP STDOUT] ${line}`);
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ log: line }));
                }
            }

            boundary = buffer.indexOf('\n');
        }
    });


    process.stderr.on('data', (data) => {
        log(`[CPP STDERR] ${data.toString().trim()}`);
    });

    process.on('error', (err) => {
        log(`[CPP ERROR] Failed to start process: ${err.message}`);
    });

    process.on('exit', (code, signal) => {
        log(`[CPP EXIT] Process exited with code ${code}, signal ${signal}`);
    });

    process.on('close', (code) => {
        log(`[CPP CLOSE] Process closed with code ${code}`);
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ status: 'stopped', code }));
        }
    });

    ws.on('message', (message) => {
        const msg = message.toString();
        log(`[WS MESSAGE] Received: ${msg}`);
        if (process.stdin) {
            log(`[CPP STDIN] Writing: ${msg}`);
            process.stdin.write(msg + '\n');
        } else {
            log('[CPP STDIN] Error: stdin not available');
        }
    });

    ws.on('close', () => {
        log('Client disconnected (WebSocket), killing C++ process');
        process.kill();
    });

    ws.on('error', (err) => {
        log(`[WS ERROR] ${err.message}`);
    });
});

const PORT = 8080;
server.listen(PORT, () => {
    log(`Server listening on http://localhost:${PORT}`);
});

