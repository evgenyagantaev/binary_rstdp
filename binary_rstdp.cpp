#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

// --- Logging System ---
std::mutex g_log_mutex;
void log_to_file(const std::string &message) {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::ofstream log_file("backend.log", std::ios_base::app);
  if (log_file.is_open()) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm buf;
    localtime_s(&buf, &in_time_t);
    log_file << "[" << std::put_time(&buf, "%Y-%m-%d %H:%M:%S") << "] "
             << message << std::endl;
  }
}

// --- Neuron parameters ---
const int V_THRESH = 2;
const int V_REST = 0;
const int REFRACTORY_PERIOD = 1;
const int MEMBRANE_DECAY_PERIOD = 750;

// --- Synapse / R-STDP parameters ---
const int CONFIDENCE_MAX = 5;
const int CONFIDENCE_THR = 1;
const int SPIKE_TRACE_WINDOW = 10;
const int ELIGIBILITY_TRACE_WINDOW = 100;
const int CONFIDENCE_LEAK_PERIOD = 5300;
const int REINFORCEMENT_INERTIA_PERIOD = 10;
const int PRUNING_PERIOD = 150;

// Simulation Constants
const int WORLD_SIZE = 60;
const int BRAIN_SIZE = 36; // 4 sensors + 2 motors + 30 hidden
const int CONSTANT_REWARD_DURATION = 0;
const double CONNECTION_DENSITY = 0.1;
const int CONFIDENCE_INIT_LOW = CONFIDENCE_THR;
const int CONFIDENCE_INIT_HIGH = CONFIDENCE_MAX;
const int RANDOM_ACTIVITY_COUNT = 1;
const int RANDOM_ACTIVITY_PERIOD = 5;

// --- Data structures ---

struct DigitalSynapse {
  int target_neuron_idx;
  int confidence;
  bool active;

  int ltp_timer;
  int ltd_timer;

  bool eligible_for_LTP;
  bool eligible_for_LTD;
  int eligibility_ltp_timer;
  int eligibility_ltd_timer;
  int confidence_leak_timer;
  bool highlighted;

  bool reward_acceptor;
  bool penalty_acceptor;
  int reward_inertia_counter;
  int penalty_inertia_counter;

  int ticks_since_ltp;
  bool plastic;

  DigitalSynapse(int target, int init_conf = 1, bool _plastic = true)
      : target_neuron_idx(target), confidence(init_conf),
        active(init_conf >= CONFIDENCE_THR), ltp_timer(0), ltd_timer(0),
        eligible_for_LTP(false), eligible_for_LTD(false),
        eligibility_ltp_timer(0), eligibility_ltd_timer(0),
        confidence_leak_timer(CONFIDENCE_LEAK_PERIOD), highlighted(false),
        reward_acceptor(true), penalty_acceptor(true),
        reward_inertia_counter(0), penalty_inertia_counter(0),
        ticks_since_ltp(0), plastic(_plastic) {}
};

struct DigitalNeuron {
  int id;
  int voltage;
  int refractory_timer;
  bool spiked_this_step;
  int input_buffer;
  int leak_timer;

  // History for causal tracing
  struct Contribution {
    int from_row;
    int syn_idx;
  };
  std::vector<Contribution> next_contributors;
  std::vector<std::vector<Contribution>> contrib_history;
  std::vector<bool> spike_history;
  static const int MAX_HIST = 32;

  DigitalNeuron(int _id)
      : id(_id), voltage(0), refractory_timer(0), spiked_this_step(false),
        input_buffer(0), leak_timer(MEMBRANE_DECAY_PERIOD) {
    contrib_history.resize(MAX_HIST);
    spike_history.resize(MAX_HIST, false);
  }
};

class SpikingNet {
public:
  std::vector<DigitalNeuron> neurons;
  std::vector<std::vector<DigitalSynapse>> connections;
  int global_tick;

  SpikingNet(int num_neurons) : global_tick(0) {
    for (int i = 0; i < num_neurons; ++i) {
      neurons.emplace_back(i);
      connections.push_back({});
    }
  }

  void connect_randomly(double density, std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    std::uniform_int_distribution<int> conf_dist(CONFIDENCE_INIT_LOW,
                                                 CONFIDENCE_INIT_HIGH);

    // 1. Deterministic Connections (Sensors and Motors)
    // Sensor 0 -> 6 (Food-L)
    connections[0].emplace_back(6, CONFIDENCE_MAX, false);
    // Sensor 1 -> 7 (Food-R)
    connections[1].emplace_back(7, CONFIDENCE_MAX, false);
    // Sensor 2 -> 8 (Danger-L)
    connections[2].emplace_back(8, CONFIDENCE_MAX, false);
    // Sensor 3 -> 9 (Danger-R)
    connections[3].emplace_back(9, CONFIDENCE_MAX, false);
    // 10 -> Motor 4
    connections[10].emplace_back(4, CONFIDENCE_MAX, false);
    // 11 -> Motor 5
    connections[11].emplace_back(5, CONFIDENCE_MAX, false);

    // 2. Random Hidden-to-Hidden Connections (Neurons 6 to 35)
    std::vector<int> hidden_indices;
    for (int i = 6; i < (int)neurons.size(); ++i)
      hidden_indices.push_back(i);

    for (int i : hidden_indices) {
      for (int j : hidden_indices) {
        if (i == j)
          continue;

        // Constraint 1: First layer (6-11) can't connect to each other
        if (i >= 6 && i <= 11 && j >= 6 && j <= 11)
          continue;

        // Constraint 2: Neurons 6, 7, 8, 9 can only HAVE outgoing connections
        // (no incoming besides sensor)
        if (j == 6 || j == 7 || j == 8 || j == 9)
          continue;

        // Constraint 3: Neurons 10 and 11 can only HAVE incoming connections
        // (no outgoing besides motor)
        if (i == 10 || i == 11)
          continue;

        if (dist(rng) < density) {
          int init_conf = conf_dist(rng);
          connections[i].emplace_back(j, init_conf);
        }
      }
    }

    // Ensure neurons 10 and 11 always have at least one input
    for (int target : {10, 11}) {
      bool has_input = false;
      for (const auto &row : connections) {
        for (const auto &syn : row) {
          if (syn.target_neuron_idx == target) {
            has_input = true;
            break;
          }
        }
        if (has_input)
          break;
      }
      if (!has_input) {
        // Add one from a hidden neuron (12-29) to avoid constraints
        std::uniform_int_distribution<int> src_dist(12, 29);
        connections[src_dist(rng)].emplace_back(target, CONFIDENCE_THR);
      }
    }
  }

  // sensory_input: number of input spikes per neuron at this timestep
  // reward_active: true if global reward signal is present
  // penalty_active: true if global penalty signal is present
  void step(const std::vector<int> &sensory_input, bool reward_active,
            bool penalty_active, std::mt19937 &rng) {
    ++global_tick;
    // 0. Update highlights and tick
    for (auto &row : connections) {
      for (auto &syn : row)
        syn.highlighted = false;
    }

    // 1. Update neurons
    for (auto &n : neurons) {
      bool potential_changed = false;
      n.spiked_this_step = false;

      if (n.refractory_timer > 0) {
        n.refractory_timer--;
        n.voltage = V_REST;
        n.input_buffer = 0;
        n.leak_timer = MEMBRANE_DECAY_PERIOD;
      } else {
        // Check for inputs
        if (n.input_buffer > 0)
          potential_changed = true;
        if (n.id < static_cast<int>(sensory_input.size()) &&
            sensory_input[n.id] > 0)
          potential_changed = true;

        n.voltage += n.input_buffer;
        if (n.id < static_cast<int>(sensory_input.size())) {
          if (sensory_input[n.id] > 0)
            n.voltage += V_THRESH;
        }
        n.input_buffer = 0;

        if (n.voltage >= V_THRESH) {
          n.voltage = V_REST;
          n.spiked_this_step = true;
          n.refractory_timer = REFRACTORY_PERIOD;
          potential_changed = true;
        }

        if (potential_changed) {
          n.leak_timer = MEMBRANE_DECAY_PERIOD;
        } else if (n.voltage > V_REST) {
          n.leak_timer--;
          if (n.leak_timer <= 0) {
            n.voltage--;
            n.leak_timer = MEMBRANE_DECAY_PERIOD;
          }
        } else {
          n.leak_timer = MEMBRANE_DECAY_PERIOD;
        }
      }
    }

    // 2. Propagate and evaluate plastic synapses
    DigitalSynapse *worst_syn = nullptr;
    int max_inactive = -1;
    int worst_pre_idx = -1;

    for (int i = 0; i < static_cast<int>(neurons.size()); ++i) {
      auto &synapses = connections[i];
      for (auto &syn : synapses) {
        if (neurons[i].spiked_this_step && syn.active) {
          neurons[syn.target_neuron_idx].input_buffer += 1;
          neurons[syn.target_neuron_idx].next_contributors.push_back(
              {i, (int)(&syn - &synapses[0])});
        }

        if (syn.plastic) {
          syn.ticks_since_ltp++;

          // Reset inactivity counter on LTP attempt (Reward + Eligible)
          if (reward_active && syn.reward_acceptor && syn.eligible_for_LTP) {
            syn.ticks_since_ltp = 0;
          }

          if (syn.ticks_since_ltp > max_inactive) {
            max_inactive = syn.ticks_since_ltp;
            worst_syn = &syn;
            worst_pre_idx = i;
          }

          // Decay timers
          if (syn.ltp_timer > 0)
            syn.ltp_timer--;
          if (syn.ltd_timer > 0)
            syn.ltd_timer--;

          // Reinforcement Inertia counters
          if (syn.reward_inertia_counter > 0) {
            syn.reward_inertia_counter--;
            if (syn.reward_inertia_counter == 0)
              syn.reward_acceptor = true;
          }
          if (syn.penalty_inertia_counter > 0) {
            syn.penalty_inertia_counter--;
            if (syn.penalty_inertia_counter == 0)
              syn.penalty_acceptor = true;
          }

          if (syn.eligibility_ltp_timer > 0) {
            syn.eligibility_ltp_timer--;
            if (syn.eligibility_ltp_timer == 0)
              syn.eligible_for_LTP = false;
          }
          if (syn.eligibility_ltd_timer > 0) {
            syn.eligibility_ltd_timer--;
            if (syn.eligibility_ltd_timer == 0)
              syn.eligible_for_LTD = false;
          }

          // Trace creation
          if (neurons[i].spiked_this_step) {
            syn.ltp_timer = SPIKE_TRACE_WINDOW;
            if (syn.ltd_timer > 0) {
              syn.eligible_for_LTD = true;
              syn.eligibility_ltd_timer = ELIGIBILITY_TRACE_WINDOW;
            }
          }

          if (neurons[syn.target_neuron_idx].spiked_this_step) {
            syn.ltd_timer = SPIKE_TRACE_WINDOW;
            if (syn.ltp_timer > 0) {
              syn.eligible_for_LTP = true;
              syn.eligibility_ltp_timer = ELIGIBILITY_TRACE_WINDOW;
            }
          }

          // Learning
          if (reward_active && syn.reward_acceptor) {
            bool modified = false;
            if (syn.eligible_for_LTP && syn.confidence < CONFIDENCE_MAX) {
              syn.confidence++;
              syn.active = (syn.confidence >= CONFIDENCE_THR);
              syn.eligible_for_LTP = false;
              syn.eligibility_ltp_timer = 0;
              syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
              modified = true;
            }
            if (!modified && syn.eligible_for_LTD && syn.confidence > 0) {
              syn.confidence--;
              syn.active = (syn.confidence >= CONFIDENCE_THR);
              syn.eligible_for_LTD = false;
              syn.eligibility_ltd_timer = 0;
              syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
              modified = true;
            }
            if (modified) {
              // Block penalty for a while
              syn.penalty_acceptor = false;
              syn.penalty_inertia_counter = REINFORCEMENT_INERTIA_PERIOD;
            }
          } else if (penalty_active && syn.penalty_acceptor) {
            bool modified = false;
            if (syn.eligible_for_LTP && syn.confidence > 0) {
              syn.confidence--;
              syn.active = (syn.confidence >= CONFIDENCE_THR);
              syn.eligible_for_LTP = false;
              syn.eligibility_ltp_timer = 0;
              syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
              modified = true;
            }
            if (modified) {
              // Block reward for a while
              syn.reward_acceptor = false;
              syn.reward_inertia_counter = REINFORCEMENT_INERTIA_PERIOD;
            }
            // LTD + PENALTY is ignored per user request
            if (syn.eligible_for_LTD) {
              syn.eligible_for_LTD = false;
              syn.eligibility_ltd_timer = 0;
            }
          }

          // Leak
          if (syn.confidence_leak_timer > 0)
            syn.confidence_leak_timer--;
          if (syn.confidence_leak_timer == 0) {
            syn.confidence >>= 1;
            syn.active = (syn.confidence >= CONFIDENCE_THR);
            syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
          }
        }
      }
    }

    // 2.5 Pruning: Rewire the most inactive synapse periodically
    if (worst_syn && (global_tick % PRUNING_PERIOD == 0)) {
      std::vector<int> possible_targets;
      // Potential targets: ONLY Hidden (6-35). Motors (4,5) are fixed.
      for (int j = 6; j < static_cast<int>(neurons.size()); ++j) {
        if (j == worst_pre_idx)
          continue;
        // Constraint 1: First layer (6-11) can't connect to each other
        if (worst_pre_idx >= 6 && worst_pre_idx <= 11 && j >= 6 && j <= 11)
          continue;

        // Constraint 2: Neurons 6, 7, 8, 9 can only HAVE outgoing connections
        if (j == 6 || j == 7 || j == 8 || j == 9)
          continue;

        // Ensure no duplicate
        bool exists = false;
        for (const auto &s : connections[worst_pre_idx]) {
          if (s.target_neuron_idx == j) {
            exists = true;
            break;
          }
        }
        if (!exists)
          possible_targets.push_back(j);
      }

      if (!possible_targets.empty()) {
        std::uniform_int_distribution<int> t_dist(0,
                                                  possible_targets.size() - 1);
        int new_target = possible_targets[t_dist(rng)];

        // Constraint: If current target is 10 or 11, check if this is the only
        // connection
        if (worst_syn->target_neuron_idx == 10 ||
            worst_syn->target_neuron_idx == 11) {
          int count = 0;
          for (const auto &row : connections) {
            for (const auto &s : row) {
              if (s.target_neuron_idx == worst_syn->target_neuron_idx)
                count++;
            }
          }
          if (count <= 1) {
            // Can't move it away. Effectively just reset it on current target.
            new_target = worst_syn->target_neuron_idx;
          }
        }

        // Prune and Rewire
        worst_syn->target_neuron_idx = new_target;
        worst_syn->confidence = 1; // Start fresh
        worst_syn->active = (worst_syn->confidence >= CONFIDENCE_THR);
        worst_syn->ticks_since_ltp = 0; // Reset timer

        // Reset plastic state
        worst_syn->ltp_timer = 0;
        worst_syn->ltd_timer = 0;
        worst_syn->eligible_for_LTP = false;
        worst_syn->eligible_for_LTD = false;
        worst_syn->eligibility_ltp_timer = 0;
        worst_syn->eligibility_ltd_timer = 0;
        worst_syn->reward_acceptor = true;
        worst_syn->penalty_acceptor = true;
      }
    }

    // 3. Causal Tracing from Motor Neurons
    for (int m = 4; m <= 5; ++m) {
      trace_causal_chain(m);
    }

    // 4. Finally Shift history for next step
    for (auto &n : neurons) {
      for (int h = DigitalNeuron::MAX_HIST - 1; h > 0; --h) {
        n.contrib_history[h] = std::move(n.contrib_history[h - 1]);
        n.spike_history[h] = n.spike_history[h - 1];
      }
      n.contrib_history[0] = std::move(n.next_contributors);
      n.spike_history[0] = n.spiked_this_step;
      n.next_contributors.clear();
    }
  }

  void trace_causal_chain(int motor_idx) {
    if (!neurons[motor_idx].spiked_this_step)
      return;

    // We trace back to show what CAUSED the current motor spike.
    // Limit depth to 12 for cleaner visualization (enough for direct paths).
    const int MAX_TRACE = 12;
    std::vector<std::vector<bool>> visited(
        MAX_TRACE + 1, std::vector<bool>(neurons.size(), false));

    struct Pending {
      int idx;
      int depth;
    };
    std::vector<Pending> stack;

    stack.push_back({motor_idx, 0});
    visited[0][motor_idx] = true;

    while (!stack.empty()) {
      Pending p = stack.back();
      stack.pop_back();

      if (p.depth >= MAX_TRACE)
        continue;

      // Spike at depth 'p.depth' was caused by signals in
      // contrib_history[p.depth] (these signals were sent at time T - p.depth -
      // 1).
      for (const auto &c : neurons[p.idx].contrib_history[p.depth]) {
        if (c.from_row >= 0 && c.from_row < (int)connections.size()) {
          connections[c.from_row][c.syn_idx].highlighted = true;

          int next_depth = p.depth + 1;
          // The sender must have spiked at T - next_depth.
          // Before the end-of-step shift, spike_history[p.depth] contains T -
          // (p.depth + 1).
          if (next_depth <= MAX_TRACE) {
            if (neurons[c.from_row].spike_history[p.depth] &&
                !visited[next_depth][c.from_row]) {
              visited[next_depth][c.from_row] = true;
              stack.push_back({c.from_row, next_depth});
            }
          }
        }
      }
    }
  }
};

// --- World Simulation ---
enum TargetType { NONE, FOOD, DANGER };

struct WorldUpdateResult {
  bool reward;
  bool penalty;
};

struct World {
  int size;
  int agent_pos;
  int target_pos;
  TargetType target_type;
  int target_timer;
  int food_eaten;
  int danger_hit;
  std::mt19937 rng;

  World()
      : size(WORLD_SIZE), agent_pos(WORLD_SIZE / 2), target_type(NONE),
        target_timer(0), food_eaten(0), danger_hit(0), rng(42) {}

  void spawn_target() {
    std::uniform_int_distribution<int> type_dist(0, 2);
    std::uniform_int_distribution<int> time_dist(3000, 5000);

    int choice = type_dist(rng);
    target_timer = time_dist(rng);

    // Reset agent to center on target change
    agent_pos = size / 2;

    if (choice == 2) {
      target_type = NONE;
      target_timer /= 3; // NONE phase is 3x shorter
    } else {
      target_type = (choice == 0) ? FOOD : DANGER;
      // Randomly spawn at extreme left (0) or extreme right (size-1)
      std::uniform_int_distribution<int> edge_dist(0, 1);
      target_pos = (edge_dist(rng) == 0) ? 0 : size - 1;
    }
  }

  std::vector<int> get_sensors() {
    std::vector<int> sensors(4, 0);
    if (target_type == NONE)
      return sensors;

    bool is_left = target_pos < agent_pos;
    if (target_type == FOOD) {
      sensors[0] = is_left ? 1 : 0;
      sensors[1] = !is_left ? 1 : 0;
    } else {
      sensors[2] = is_left ? 1 : 0;
      sensors[3] = !is_left ? 1 : 0;
    }
    return sensors;
  }

  WorldUpdateResult update(bool move_left, bool move_right) {
    if (target_timer <= 0)
      spawn_target();

    int prev_dist = -1;
    if (target_type != NONE) {
      prev_dist = std::abs(agent_pos - target_pos);
    } else {
      // Force move to middle when no target exists
      int mid = size / 2;
      if (agent_pos < mid)
        agent_pos++;
      else if (agent_pos > mid)
        agent_pos--;
    }

    if (move_left)
      agent_pos--;
    if (move_right)
      agent_pos++;

    WorldUpdateResult res = {false, false};
    if (target_type != NONE) {
      int curr_dist = std::abs(agent_pos - target_pos);

      if (target_type == FOOD) {
        if (curr_dist < prev_dist)
          res.reward = true;
        else if (curr_dist > prev_dist)
          res.penalty = true;
      } else if (target_type == DANGER) {
        if (curr_dist > prev_dist)
          res.reward = true;
        else if (curr_dist < prev_dist)
          res.penalty = true;
      }

      if (curr_dist == 0) {
        if (target_type == FOOD) {
          food_eaten++;
          res.reward = true;
          res.penalty = false; // Reward takes precedence
        } else {
          danger_hit++;
          res.penalty = true;
          res.reward = false;
        }
        // Reset agent to center without removing target
        agent_pos = size / 2;
      }
    }

    if (target_timer > 0) {
      target_timer--;
      if (target_timer <= 0)
        target_type = NONE;
    }

    return res;
  }
};

// --- SIMULATION CONTROL ---
std::atomic<bool> g_paused(true); // Start paused
std::atomic<bool> g_reset(false);
std::atomic<int> g_delay_ms(500);
std::atomic<bool> g_running(true);

void input_listener() {
  std::string cmd;
  while (std::cin >> cmd) {
    log_to_file("Received command: " + cmd);
    std::cerr << "[CPP] Received command: " << cmd << std::endl;
    if (cmd == "stop") {
      g_running = false;
      break;
    } else if (cmd == "pause") {
      g_paused = true;
    } else if (cmd == "resume" || cmd == "start") {
      g_paused = false;
    } else if (cmd == "reset") {
      g_reset = true;
    } else if (cmd == "speed") {
      int val;
      if (std::cin >> val) {
        log_to_file("Speed value received: " + std::to_string(val));
        std::cerr << "[CPP] Speed value: " << val << std::endl;
        if (val < 0)
          val = 0;
        g_delay_ms = val;
      }
    }
  }
}

void print_json_state(const SpikingNet &net, const World &world, int tick,
                      bool reward, bool penalty, int reward_sum,
                      int penalty_sum, int food_time, int danger_time) {
  std::cout << "{";
  std::cout << "\"reward\":" << (reward ? "true" : "false") << ",";
  std::cout << "\"penalty\":" << (penalty ? "true" : "false") << ",";
  std::cout << "\"reward_sum\":" << reward_sum << ",";
  std::cout << "\"penalty_sum\":" << penalty_sum << ",";
  std::cout << "\"food_time\":" << food_time << ",";
  std::cout << "\"danger_time\":" << danger_time << ",";
  std::cout << "\"t\":" << tick << ",";

  std::cout << "\"world\":{";
  std::cout << "\"agent\":" << world.agent_pos << ",";
  std::cout << "\"target\":" << world.target_pos << ",";
  std::cout << "\"type\":" << world.target_type << ",";
  std::cout << "\"food\":" << world.food_eaten << ",";
  std::cout << "\"danger\":" << world.danger_hit << ",";
  std::cout << "\"dist\":"
            << ((world.target_type != NONE)
                    ? std::abs(world.agent_pos - world.target_pos)
                    : 0);
  std::cout << "},";

  std::cout << "\"neurons\":[";
  for (size_t i = 0; i < net.neurons.size(); ++i) {
    const auto &n = net.neurons[i];
    std::cout << "{\"id\":" << n.id << ",\"v\":" << n.voltage
              << ",\"s\":" << (n.spiked_this_step ? "true" : "false") << "}";
    if (i < net.neurons.size() - 1)
      std::cout << ",";
  }
  std::cout << "],";

  std::cout << "\"synapses\":[";
  bool first = true;
  for (size_t i = 0; i < net.connections.size(); ++i) {
    for (const auto &syn : net.connections[i]) {
      if (!first)
        std::cout << ",";
      std::cout << "{\"s\":" << i << ",\"t\":" << syn.target_neuron_idx
                << ",\"c\":" << syn.confidence
                << ",\"a\":" << (syn.active ? "true" : "false")
                << ",\"b\":" << (syn.highlighted ? "1" : "0") << "}";
      first = false;
    }
  }
  std::cout << "]";
  std::cout << "}" << std::endl;
}

int main() {
  try {
    log_to_file("Process started");

    // Launch input listener thread
    std::thread input_thread(input_listener);
    input_thread.detach();

    while (g_running) {
      log_to_file("Entering simulation loop");

      // --- Initialization ---
      SpikingNet brain(BRAIN_SIZE);
      std::mt19937 rng(static_cast<unsigned int>(
          std::chrono::system_clock::now().time_since_epoch().count()));
      brain.connect_randomly(CONNECTION_DENSITY, rng);
      World world;

      // Reset flag cleared
      g_reset = false;

      // Reward/Penalty logic state
      bool current_reward = true; // Initially true during constant duration
      bool current_penalty = false;
      int reward_sum = 0;
      int penalty_sum = 0;
      int food_time = 0;
      int danger_time = 0;

      // --- Simulation Loop ---
      for (int t = 0; g_running && !g_reset; ++t) {
        // Output state FIRST (so we see initial state or current state)
        print_json_state(brain, world, t, current_reward, current_penalty,
                         reward_sum, penalty_sum, food_time, danger_time);

        // Wait if paused or for speed control
        int delay = g_delay_ms;
        if (g_paused) {
          while (g_paused && g_running && !g_reset) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
          delay = 0;
        }
        if (delay > 0)
          std::this_thread::sleep_for(std::chrono::milliseconds(delay));

        if (!g_running || g_reset)
          break;

        // 1. Sensors
        auto sensors = world.get_sensors();
        std::vector<int> net_input(BRAIN_SIZE, 0);
        for (int i = 0; i < 4; ++i) {
          net_input[i] = sensors[i];
        }

        // 1.5. Random wandering activity
        if (RANDOM_ACTIVITY_PERIOD > 0 && t % RANDOM_ACTIVITY_PERIOD == 0) {
          std::uniform_int_distribution<int> rand_neuron_dist(6,
                                                              BRAIN_SIZE - 1);
          for (int i = 0; i < RANDOM_ACTIVITY_COUNT; ++i) {
            int rand_idx = rand_neuron_dist(rng);
            net_input[rand_idx]++;
          }
        }

        // 5. Brain Step
        bool force_reward = false;
        brain.step(net_input, force_reward || current_reward,
                   (!force_reward) && current_penalty, rng);

        // 3. Motors
        bool m_left = brain.neurons[4].spiked_this_step;
        bool m_right = brain.neurons[5].spiked_this_step;
        if (m_left && m_right) {
          m_left = false;
          m_right = false;
        }

        // 4. World Update
        auto res = world.update(m_left, m_right);

        // 5. Update Reward/Penalty for NEXT step
        current_reward = res.reward;
        current_penalty = res.penalty;
        if (current_reward)
          reward_sum++;
        if (current_penalty)
          penalty_sum++;
        if (world.target_type == FOOD)
          food_time++;
        else if (world.target_type == DANGER)
          danger_time++;
      }

      if (g_reset) {
        log_to_file("Simulation reset triggered");
      }
    }
    log_to_file("Process exiting normally");
  } catch (const std::exception &e) {
    log_to_file("CRITICAL ERROR: " + std::string(e.what()));
    std::cerr << "[CPP ERROR] " << e.what() << std::endl;
    return 1;
  } catch (...) {
    log_to_file("CRITICAL ERROR: Unknown exception");
    std::cerr << "[CPP ERROR] Unknown exception" << std::endl;
    return 1;
  }
  return 0;
}
