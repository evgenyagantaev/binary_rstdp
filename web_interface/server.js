const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Serve static files from 'public' directory
app.use(express.static(path.join(__dirname, 'public')));

// Path to C++ executable
const EXE_PATH = path.join(__dirname, '../x64/Debug/binary_rstdp.exe');
const CMD = EXE_PATH;

console.log(`Targeting executable: ${CMD}`);

wss.on('connection', (ws) => {
    console.log('Client connected');
    console.log('Spawning C++ process...');

    let buffer = '';
    const process = spawn(CMD, [], {
        cwd: path.join(__dirname, '../')
    });

    process.stdout.on('data', (data) => {
        buffer += data.toString();

        let boundary = buffer.indexOf('\n');
        while (boundary !== -1) {
            const line = buffer.substring(0, boundary).trim();
            buffer = buffer.substring(boundary + 1);

            if (line.startsWith('{') && line.endsWith('}')) {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(line);
                }
            } else if (line.startsWith('[WORLD]')) {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ log: line }));
                }
            } else if (line.startsWith('[CPP]')) {
                // Also forward logging from C++ stderr/stdout if it appears here (depends on C++ impl)
                console.log(line);
            }

            boundary = buffer.indexOf('\n');
        }
    });

    process.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
    });

    process.on('close', (code) => {
        console.log(`C++ process exited with code ${code}`);
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ status: 'stopped', code }));
        }
    });

    ws.on('message', (message) => {
        const msg = message.toString();
        console.log('Received from Client:', msg);
        if (process.stdin) {
            console.log('Writing to C++ stdin:', msg);
            process.stdin.write(msg + '\n');
        } else {
            console.error('C++ stdin not available');
        }
    });

    ws.on('close', () => {
        console.log('Client disconnected, killing process');
        process.kill();
    });
});

const PORT = 8080;
server.listen(PORT, () => {
    console.log(`Server started on http://localhost:${PORT}`);
});
