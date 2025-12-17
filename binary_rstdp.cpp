#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <string>

// --- Neuron parameters ---
// Any incoming spike adds +1 to membrane potential.
// Membrane potential threshold for spike = 10.
const int V_THRESH = 10;
const int V_REST = 0;
const int REFRACTORY_PERIOD = 2;
const int MEMBRANE_DECAY_PERIOD = 20;

// Global tick counter for the whole model:
// every 10th tick all neuron membrane potentials are halved.
static int GLOBAL_TICK = 0;

// --- Synapse / R-STDP parameters ---
const int CONFIDENCE_MAX = 2;
const int CONFIDENCE_THR = 2;
const int SPIKE_TRACE_WINDOW = 10;
const int ELIGIBILITY_TRACE_WINDOW = 100;
const int CONFIDENCE_LEAK_PERIOD = 1000;

// Global reward flag for learning
static bool REWARD = true;

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

    DigitalSynapse(int target, int init_conf = 1)
        : target_neuron_idx(target),
          confidence(init_conf),
          active(init_conf >= CONFIDENCE_THR),
          ltp_timer(0),
          ltd_timer(0),
          eligible_for_LTP(false),
          eligible_for_LTD(false),
          eligibility_ltp_timer(0),
          eligibility_ltd_timer(0) {}
};

struct DigitalNeuron {
    int id;
    int voltage;
    int refractory_timer;
    bool spiked_this_step;
    int input_buffer;

    DigitalNeuron(int _id)
        : id(_id),
          voltage(0),
          refractory_timer(0),
          spiked_this_step(false),
          input_buffer(0) {}
};

class SpikingNet {
public:
    std::vector<DigitalNeuron> neurons;
    std::vector<std::vector<DigitalSynapse>> connections;

    SpikingNet(int num_neurons) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(i);
            connections.push_back({});
        }
    }

    void add_synapse(int pre_idx, int post_idx, int init_conf) {
        connections[pre_idx].emplace_back(post_idx, init_conf);
    }

    void clear_connections() {
        for (auto& conn : connections) {
            conn.clear();
        }
    }

    // sensory_input: number of input spikes per neuron at this timestep
    void step(const std::vector<int>& sensory_input) {
        // Advance global time and decide whether to apply decay this step
        ++GLOBAL_TICK;
        const bool apply_decay = (GLOBAL_TICK % MEMBRANE_DECAY_PERIOD == 0);

        // 1. Update neurons
        for (auto& n : neurons) {
            n.spiked_this_step = false;

            // Every 10 ticks: halve membrane potential (integer shift)
            if (apply_decay) {
                n.voltage >>= 1;
            }

            if (n.refractory_timer > 0) {
                n.refractory_timer--;
                n.voltage = V_REST;
                n.input_buffer = 0;
            } else {
                // integrate incoming spikes
                n.voltage += n.input_buffer;
                if (n.id < static_cast<int>(sensory_input.size())) {
                    n.voltage += sensory_input[n.id];
                }
                n.input_buffer = 0;

                // threshold
                if (n.voltage >= V_THRESH) {
                    n.voltage = V_REST;
                    n.spiked_this_step = true;
                    n.refractory_timer = REFRACTORY_PERIOD;
                }
            }
        }

        // 2. Propagate spikes and apply R-STDP
        for (int i = 0; i < static_cast<int>(neurons.size()); ++i) {
            auto& synapses = connections[i];
            for (auto& syn : synapses) {
                // spike propagation: each presynaptic spike adds +1
                if (neurons[i].spiked_this_step && syn.active) {
                    neurons[syn.target_neuron_idx].input_buffer += 1;
                }

                // decay spike traces
                if (syn.ltp_timer > 0) syn.ltp_timer--;
                if (syn.ltd_timer > 0) syn.ltd_timer--;

                // decay eligibility traces
                if (syn.eligibility_ltp_timer > 0) {
                    syn.eligibility_ltp_timer--;
                    if (syn.eligibility_ltp_timer == 0) {
                        syn.eligible_for_LTP = false;
                    }
                }
                if (syn.eligibility_ltd_timer > 0) {
                    syn.eligibility_ltd_timer--;
                    if (syn.eligibility_ltd_timer == 0) {
                        syn.eligible_for_LTD = false;
                    }
                }

                // Pre-spike: start pre-trace and check for LTD eligibility
                if (neurons[i].spiked_this_step) {
                    syn.ltp_timer = SPIKE_TRACE_WINDOW;
                    if (syn.ltd_timer > 0) {
                        syn.eligible_for_LTD = true;
                        syn.eligibility_ltd_timer = ELIGIBILITY_TRACE_WINDOW;
                    }
                }

                // Post-spike: start post-trace and check for LTP eligibility
                if (neurons[syn.target_neuron_idx].spiked_this_step) {
                    syn.ltd_timer = SPIKE_TRACE_WINDOW;
                    if (syn.ltp_timer > 0) {
                        syn.eligible_for_LTP = true;
                        syn.eligibility_ltp_timer = ELIGIBILITY_TRACE_WINDOW;
                    }
                }

                // Reward-modulated learning (gated by global REWARD)
                if (REWARD) {
                    if (syn.eligible_for_LTP && syn.confidence < CONFIDENCE_MAX) {
                        syn.confidence++;
                        syn.eligible_for_LTP = false;
                        syn.eligibility_ltp_timer = 0;
                        syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
                    }
                    if (syn.eligible_for_LTD && syn.confidence > 0) {
                        syn.confidence--;
                        syn.eligible_for_LTD = false;
                        syn.eligibility_ltd_timer = 0;
                        syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
                    }
                }

                // Slow synaptic leak: every CONFIDENCE_LEAK_PERIOD ticks halve confidence
                if (syn.confidence_leak_timer > 0) syn.confidence_leak_timer--;
                if (syn.confidence_leak_timer == 0) {
                    syn.confidence >>= 1;
                    syn.confidence_leak_timer = CONFIDENCE_LEAK_PERIOD;
                }

                syn.active = (syn.confidence >= CONFIDENCE_THR);
            }
        }
    }
};

// --- Tests ---

// 1. LIF neuron dynamics
// - input spike probability = 0.5
// - simulate until 3rd output spike, then 10 extra timesteps
void test_LIF_Dynamics() {
    std::cout << "\n=== TEST 1: LIF Neuron Dynamics ===\n";
    SpikingNet net(1); // single neuron

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::cout << "T\tInput\tVolt\tSpike\n";

    int spike_count = 0;
    int t = 0;
    const int extra_steps_after_third_spike = 10;
    int steps_after_third = 0;
    bool third_spike_seen = false;

    // Safety limit to avoid infinite loop in pathological cases
    const int MAX_STEPS = 100000;

    while (t < MAX_STEPS) {
        std::vector<int> input(1, 0);

        // Bernoulli(p=0.5) input spike
        if (dist(rng) < 0.5) {
            input[0] = 1;
        }

        net.step(input);

        std::cout << t << "\t" << input[0] << "\t"
                  << net.neurons[0].voltage << "\t"
                  << (net.neurons[0].spiked_this_step ? "BOOM!" : ".")
                  << std::endl;

        if (!third_spike_seen) {
            if (net.neurons[0].spiked_this_step) {
                ++spike_count;
                if (spike_count == 3) {
                    third_spike_seen = true;
                }
            }
        } else {
            ++steps_after_third;
            if (steps_after_third >= extra_steps_after_third_spike) {
                break;
            }
        }

        ++t;
    }
}

// 2. Extended STDP mechanics test with random spikes and constant reward
void test_STDP_Mechanics_001() {
    std::cout << "\n=== TEST 2.1: STDP Mechanics 001 (random spikes, constant reward, conduction check) ===\n";

    // Scenario 1: random pre/post spikes with p=0.3 and reward always on
    {
        std::cout << "\nScenario 1: random spikes with constant reward\n";

        GLOBAL_TICK = 0;

        SpikingNet net(2);          // neuron 0 -> neuron 1
        net.add_synapse(0, 1, 1);   // start around threshold so synapse can switch on/off

        std::mt19937 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        auto& syn = net.connections[0][0];
        int prev_conf = syn.confidence;
        bool prev_active = syn.active;

        std::cout << "t\tin0\tin1\tV0\tV1\tspk0\tspk1\tltp\tltd\tE_LTP\tE_LTD\tconf\tactive\n";

        const int T_MAX = 5000;
        for (int t = 0; t < T_MAX; ++t) {
            std::vector<int> input(2, 0);

            // Bernoulli(p=0.5) spikes on pre and post inputs
            if (dist(rng) < 0.5) input[0] = 1;
            if (dist(rng) < 0.5) input[1] = 1;

            // Reward is always on so that confidence reacts immediately when eligible
            net.step(input);

            std::cout << t << '\t'
                      << input[0] << '\t'
                      << input[1] << '\t'
                      << net.neurons[0].voltage << '\t'
                      << net.neurons[1].voltage << '\t'
                      << net.neurons[0].spiked_this_step << '\t'
                      << net.neurons[1].spiked_this_step << '\t'
                      << syn.ltp_timer << '\t'
                      << syn.ltd_timer << '\t'
                      << syn.eligible_for_LTP << '\t'
                      << syn.eligible_for_LTD << '\t'
                      << syn.confidence << '\t'
                      << syn.active
                      << std::endl;

            if (net.neurons[0].spiked_this_step) {
                std::cout << "  [Spike] pre neuron (0) spiked at t=" << t << "\n";
            }
            if (net.neurons[1].spiked_this_step) {
                std::cout << "  [Spike] post neuron (1) spiked at t=" << t << "\n";
            }

            if (syn.confidence != prev_conf) {
                std::cout << "  [Synapse] confidence changed: " << prev_conf
                          << " -> " << syn.confidence
                          << " at t=" << t << "\n";
                prev_conf = syn.confidence;
            }
            if (syn.active != prev_active) {
                std::cout << "  [Synapse] active flag changed: " << prev_active
                          << " -> " << syn.active
                          << " at t=" << t << "\n";
                prev_active = syn.active;
            }
        }
    }
    
}


int main() {
    // seed for any legacy rand()-based code, though tests use std::mt19937
    srand(42);

    test_LIF_Dynamics();
    test_STDP_Mechanics_001();

    return 0;
}
