# Binary R-STDP: Spiking Neural Network Agent

![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![C++](https://img.shields.io/badge/C++-17-orange.svg)

An experimental Open Source project implementing a biologically-inspired **Spiking Neural Network (SNN)** controlled by **Reward-modulated Spike-Timing-Dependent Plasticity (R-STDP)**. The project features a high-performance C++ simulation engine and a real-time web-based visualizer.

The goal is to evolve an autonomous agent (represented as a 🐛 worm) in a 1D world that learns to navigate towards food (🍎) and avoid danger (☠️) through pure reinforcement and synaptic evolution.

## 🧠 Core Concepts

### 1. Spiking Neural Network (SNN)
The agent's "brain" consists of 36 digital neurons:
- **Sensors (0-3):** Detect food and danger on the left and right.
- **Motors (4-5):** Control movement (Left/Right).
- **Hidden Layer (6-35):** Process signals and evolve connectivity.

### 2. R-STDP Learning Rule
Learning is driven by a modified **4-factor R-STDP** algorithm:
- **Pre-post timing:** STDP traces record temporal correlations.
- **Global Reward/Penalty:** Environmental feedback modulates synaptic "confidence".
- **Reinforcement Inertia:** A stabilization mechanism that prevents synapses from rapidly oscillating between states immediately after a significant update.

### 3. Synaptic Pruning & Rewiring
To prevent the network from getting stuck in local minima, the system implements **Dynamic Synaptic Pruning**:
- Every 150 ticks, the most "inactive" synapse (the one that hasn't participated in a rewarded pathway for the longest time) is "pruned".
- It is then randomly rewired to a new target within the hidden layer, facilitating continuous exploration of new neural architectures.

## 🛠 Tech Stack

- **Backend:** C++17 (Simulation engine, high-concurrency loops).
- **Frontend:** HTML5, CSS3, Vanilla JavaScript (Canvas-based real-time rendering).
- **Communication:** WebSockets (JSON-based state synchronization).
- **Server:** Node.js (Mediates between C++ process and the web browser).

## 🚀 Getting Started

### Prerequisites
- Modern C++ compiler (GCC, Clang, or MSVC).
- Node.js (for the web interface).

### Installation & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/evgenyagantaev/binary_rstdp.git
   cd binary_rstdp
   ```

2. **Compile the Backend:**
   - **Easy (Windows):** Open the project in **Visual Studio** and build as a `Visual Studio Solution`. This is the recommended way for Windows users.
   - **CLI (Linux/Generic):**
     ```bash
     g++ -O3 binary_rstdp.cpp -o binary_rstdp -lpthread
     ```

3. **Install dependencies and start the UI:**
   ```bash
   cd web_interface
   npm install
   node server.js
   ```

4. **Visualize:**
   Open `http://localhost:3000` in your browser.

## 🖥 User Interface Features

- **Real-time Neural Trace:** Watch membrane potentials rise and fall.
- **Active Gradients:** Synapses glow with a **Yellow → Red** gradient indicating signal direction.
- **Causal Tracing:** When a motor neuron spikes, the interface highlights the specific ancestral pathway (up to 12 steps back) that caused that action.
- **World View:** A 1200px wide track where you can watch the agent live, reset its progress, or adjust the simulation speed.
- **Visual Feedback:** The agent "pulses" green when successfully eating food, providing immediate visual reinforcement.

## 🏗 Network Topology Rules

- **Strict Input/Output Ports:** Neurons 6-9 are dedicated entry points for sensors; 10-11 are exit points for motors.
- **Directional Constraints:** Sensors only have outgoing connections; exit nodes 10-11 primarily receive signals.
- **Stability Guarantee:** Motor neurons (10, 11) are protected from losing all inputs during the pruning process.

## 📜 License
Published under the **MIT License**. Feel free to use, modify, and contribute!

---
*Created by [Evgeny Agantaev](https://github.com/evgenyagantaev) and the Antigravity AI assistant.*
