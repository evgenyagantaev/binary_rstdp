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
const int MEMBRANE_DECAY_PERIOD = 100;

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

// Simulation
const int WORLD_SIZE = 30;
const int BRAIN_SIZE = 30; // 4 сенсора + 2 мотора + 24 скрытых
const int DURATION = 1000000;
const int CONSTANT_REWARD_DURATION = 500000;
const double CONNECTION_DENSITY = 0.6;
const int CONFIDENCE_INIT_LOW = 1; 
const int CONFIDENCE_INIT_HIGH = 2;

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
          eligibility_ltd_timer(0),
          confidence_leak_timer(CONFIDENCE_LEAK_PERIOD) {}
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

    // Случайная архитектура "куча нейронов"
    void connect_randomly(double density, std::mt19937& rng) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::uniform_int_distribution<int> conf_dist(CONFIDENCE_INIT_LOW, CONFIDENCE_INIT_HIGH);

        for (int i = 0; i < neurons.size(); ++i) {
            for (int j = 0; j < neurons.size(); ++j) {
                if (i == j) continue; // Без самовозбуждения для простоты
                if (dist(rng) < density) {
                    // Инициализируем случайной уверенностью
                    connections[i].emplace_back(j, conf_dist(rng));
                }
            }
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

// Полный вывод состояния мозга (нейроны и синапсы)
void print_brain_state(const SpikingNet& net) {
    std::cout << "\n=== Brain state after initialization ===\n";
    std::cout << "Neurons: " << net.neurons.size() << "\n";

    for (const auto& n : net.neurons) {
        std::cout << "Neuron " << n.id
                  << " | V=" << n.voltage
                  << " | refractory=" << n.refractory_timer
                  << " | spiked=" << n.spiked_this_step
                  << " | input_buf=" << n.input_buffer
                  << "\n";
    }

    std::cout << "\nConnections:\n";
    for (size_t i = 0; i < net.connections.size(); ++i) {
        const auto& synapses = net.connections[i];
        std::cout << " From neuron " << i << " (" << synapses.size() << " synapses)\n";
        for (const auto& syn : synapses) {
            std::cout << "   -> " << syn.target_neuron_idx
                      << " | conf=" << syn.confidence
                      << " | active=" << syn.active
                      << " | ltp_t=" << syn.ltp_timer
                      << " | ltd_t=" << syn.ltd_timer
                      << " | elig_LTP=" << syn.eligible_for_LTP
                      << " | elig_LTP_t=" << syn.eligibility_ltp_timer
                      << " | elig_LTD=" << syn.eligible_for_LTD
                      << " | elig_LTD_t=" << syn.eligibility_ltd_timer
                      << " | leak_t=" << syn.confidence_leak_timer
                      << "\n";
        }
    }
    std::cout << "=======================================\n\n";
}

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

// --- World Simulation ---

enum TargetType { NONE, FOOD, DANGER };

// ASCII-friendly log for world target appearance
void log_target_event(TargetType type, int target_pos, int agent_pos, int timer) {
    std::cout << "[WORLD] New target: "
              << (type == FOOD ? "FOOD" : "DANGER")
              << " at position " << target_pos
              << " (agent at " << agent_pos << ")"
              << ", lifetime " << timer << " steps"
              << std::endl;
}

struct World {
    int size;
    int agent_pos;
    int target_pos;
    TargetType target_type;
    int target_timer;
    
    // Stats
    int food_eaten;
    int danger_hit;
    int steps_survived;

    std::mt19937 rng;

    World() : size(WORLD_SIZE), agent_pos(WORLD_SIZE/2), target_type(NONE), target_timer(0), 
              food_eaten(0), danger_hit(0), steps_survived(0), rng(42) {}

    void spawn_target() {
        std::uniform_int_distribution<int> pos_dist(0, size - 1);
        std::uniform_int_distribution<int> type_dist(0, 2); // 0: FOOD, 1: DANGER, 2: NONE
        std::uniform_int_distribution<int> time_dist(2000, 3000);

        int choice = type_dist(rng);
        target_timer = time_dist(rng);

        if (choice == 2) {
            target_type = NONE;
            std::cout << "[WORLD] Period: NONE (Pause) for " << target_timer << " steps" << std::endl;
        } else {
            target_type = (choice == 0) ? FOOD : DANGER;
            // Спавним не там, где агент
            do {
                target_pos = pos_dist(rng);
            } while (target_pos == agent_pos);

            std::cout << "[WORLD] New target: "
                      << (target_type == FOOD ? "FOOD" : "DANGER")
                      << " at position " << target_pos
                      << " (agent at " << agent_pos << ")"
                      << ", lifetime " << target_timer << " steps"
                      << std::endl;
        }
    }

    // Возвращает вектор сенсоров: [FoodLeft, FoodRight, DangerLeft, DangerRight]
    // 1 - есть сигнал, 0 - нет
    std::vector<int> get_sensors() {
        std::vector<int> sensors(4, 0);
        if (target_type == NONE) return sensors;

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

    // Возвращает true, если получили награду
    bool update(bool move_left, bool move_right) {
        if (target_timer <= 0) {
            spawn_target();
        }

        int prev_dist = 0;
        int curr_dist = 0;
        
        if (target_type != NONE) {
            prev_dist = std::abs(agent_pos - target_pos);
        }
        
        if (move_left && agent_pos > 0) agent_pos--;
        if (move_right && agent_pos < size - 1) agent_pos++;

        bool reward = false;

        if (target_type != NONE) {
            curr_dist = std::abs(agent_pos - target_pos);

            // Логика награды (Градиент)
            if (target_type == FOOD) {
                if (curr_dist < prev_dist) reward = true; // Приближаемся к еде
            } else if (target_type == DANGER) {
                if (curr_dist > prev_dist) reward = true; // Убегаем от опасности
            }

            // Проверка столкновения
            if (curr_dist == 0) {
                if (target_type == FOOD) {
                    food_eaten++;
                    reward = true; // Бонус за съедение
                    std::cout << "YUMMY! ";
                } else {
                    danger_hit++;
                    reward = false; // Нет награды за смерть
                    std::cout << "OUCH! ";
                }
                target_type = NONE; // Цель исчезла
                target_timer = 0;   // Срочно спавним новое состояние
            }
        }

        // Таймер исчезновения цели или паузы
        if (target_timer > 0) {
            target_timer--;
            if (target_timer <= 0) {
                target_type = NONE;
            }
        }

        return reward;
    }
};

int main() {
    SpikingNet brain(BRAIN_SIZE);
    
    // Архитектура: 
    // 0-3: Сенсоры (Input)
    // 4-5: Моторы (Output) [4: Left, 5: Right]
    // 6-29: Hidden (Куча)
    
    // Подключаем случайно
    std::mt19937 rng(999);
    brain.connect_randomly(CONNECTION_DENSITY, rng);

    // ??????? ?????? ????????? ?????? ????? (???????????? ???????)
    print_brain_state(brain);

    World world;

    std::cout << "Starting simulation..." << std::endl;
    
    // Для статистики
    int total_reward = 0;
    int total_spikes = 0;
    int motor_spikes = 0;
    int block_size = 1000;

    for (int t = 0; t < DURATION; ++t) {
        // 1. Сенсоры -> Спайки
        auto sensors = world.get_sensors();
        
        // Маппинг сенсоров на входные нейроны (0..3)
        // Остальные 0
        std::vector<int> net_input(BRAIN_SIZE, 0);
        for(int i=0; i<4; ++i) net_input[i] = sensors[i];

        // 2. Шаг сети
        brain.step(net_input);
        for (const auto& n : brain.neurons) {
            if (n.spiked_this_step) {
                total_spikes++;
                if (n.id == 4 || n.id == 5) motor_spikes++;
            }
        }

        // 3. Чтение моторов (Нейроны 4 и 5)
        bool raw_left = brain.neurons[4].spiked_this_step;
        bool raw_right = brain.neurons[5].spiked_this_step;

        bool m_left = raw_left;
        bool m_right = raw_right;

        // Если оба спайкнули - стоим (конфликт)
        if (m_left && m_right) 
        { 
            m_left = false; 
            //m_right = false; 
        }

        // if (m_left || m_right) {
        //     std::cout << "[MOVE] Tick " << t 
        //               << " | Direction: " << (m_left ? "LEFT" : "RIGHT")
        //               << " | Raw Spikes: [M4:" << (raw_left ? "1" : "0") 
        //               << ", M5:" << (raw_right ? "1" : "0") << "]"
        //               << std::endl;
        // }

        // 4. обновляем мир и считаем награду
        bool got_reward = world.update(m_left, m_right);

        // 5. глобальный флаг REWARD для R-STDP
        // первые CONSTANT_REWARD_DURATION шагов всегда true, потом как раньше
        if (t < CONSTANT_REWARD_DURATION) {
            REWARD = true;
        } else {
            REWARD = got_reward;
        }
        if (got_reward) total_reward++;


        // Логгирование каждые N шагов
        if (t % block_size == 0 && t > 0) {
            std::cout << "Tick " << t 
                      << " | Food: " << world.food_eaten 
                      << " Danger: " << world.danger_hit
                      << " | Spikes: " << total_spikes
                      << " (M: " << motor_spikes << ")"
                      << " | Avg Reward: " << (double)total_reward / block_size 
                      << std::endl;
            
            // Сброс счетчиков для оценки динамики обучения
            world.food_eaten = 0;
            world.danger_hit = 0;
            total_reward = 0;
            total_spikes = 0;
            motor_spikes = 0;
        }
    }

    std::cout << "Done." << std::endl;
    return 0;
}

