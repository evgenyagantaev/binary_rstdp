#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// --- Глобальные настройки "Цифровой Физики" ---
const int V_THRESH = 100;       // Порог спайка
const int V_REST = 0;           // Потенциал покоя
const int V_DECAY_SHIFT = 3;    // Затухание: v -= v >> 3 (примерно 12.5% за так)
const int REFRACTORY_PERIOD = 2;// Период отдыха после спайка

const int CONFIDENCE_MAX = 100; // Макс. уверенность синапса
const int CONFIDENCE_THR = 50;  // Порог, когда синапс становится "проводящим" (вес=1)
const int TRACE_WINDOW = 50;     // Окно пластичности (сколько тактов помним спайк)

// --- Структуры ---

struct DigitalSynapse {
    int target_neuron_idx;

    // Состояние веса
    int confidence; // 0..100
    bool active;    // 1 если confidence >= CONFIDENCE_THR, иначе 0

    // STDP Трейсы (Таймеры)
    int ltp_timer; // Таймер для потенциации (усиления)
    int ltd_timer; // Таймер для депрессии (ослабления)

    DigitalSynapse(int target) : target_neuron_idx(target) {
        confidence = 20 + (rand() % 60);
        active = (confidence >= CONFIDENCE_THR);
        ltp_timer = 0;
        ltd_timer = 0;
    }
};

struct DigitalNeuron {
    int id;
    int voltage;
    int refractory_timer;
    bool spiked_this_step;

    // Входящий ток на следующий шаг (Double buffering)
    int input_buffer;

    DigitalNeuron(int _id) : id(_id), voltage(0), refractory_timer(0), spiked_this_step(false), input_buffer(0) {}
};

// --- Класс Сети (Резервуар) ---

class SpikingNet {
public:
    std::vector<DigitalNeuron> neurons;
    std::vector<std::vector<DigitalSynapse>> connections; // Списки смежности

    SpikingNet(int num_neurons) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(i);
            connections.push_back({}); // Пустой список связей для нейрона
        }
    }

    // Создание случайных связей (прокладка кабелей)
    void build_random_topology(int connections_per_neuron) {
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, neurons.size() - 1);

        for (int i = 0; i < neurons.size(); ++i) {
            for (int k = 0; k < connections_per_neuron; ++k) {
                int target = dist(rng);
                if (target != i) { // Без само-петель для простоты
                    connections[i].emplace_back(target);
                }
            }
        }
    }

    // Главный такт симуляции
    void step(const std::vector<int>& sensory_input, bool reward) {

        // 1. Обновление Нейронов (Физика)
        for (auto& n : neurons) {
            n.spiked_this_step = false;

            if (n.refractory_timer > 0) {
                n.refractory_timer--;
                n.voltage = V_REST;
                n.input_buffer = 0; // Игнорируем вход в рефрактерном периоде
            }
            else {
                // А. Затухание (Leak) - битовый сдвиг вместо умножения
                n.voltage -= (n.voltage >> V_DECAY_SHIFT);

                // !!! ДОБАВКА: Случайный шум !!!
                if ((rand() % 1000) < 5) { // 0.5% шанс случайного возбуждения
                    n.voltage += 50;
                }

                // Б. Интеграция входа (включая сенсорный, если есть для этого нейрона)
                n.voltage += n.input_buffer;
                if (n.id < sensory_input.size()) {
                    n.voltage += sensory_input[n.id];
                }
                n.input_buffer = 0; // Очистка буфера

                // В. Проверка порога
                if (n.voltage >= V_THRESH) {
                    n.voltage = V_REST;
                    n.spiked_this_step = true;
                    n.refractory_timer = REFRACTORY_PERIOD;
                }
            }
        }

        // 2. Распространение сигналов и STDP Маркировка
        for (int i = 0; i < neurons.size(); ++i) {
            auto& synapses = connections[i];

            for (auto& syn : synapses) {
                // --- Логика Передачи ---
                if (neurons[i].spiked_this_step && syn.active) {
                    // Передаем "1" (или можно усилить весом синапса, но у нас бинарный вес)
                    // Давайте передавать фиксированный заряд, например 10, чтобы быстрее копилось
                    neurons[syn.target_neuron_idx].input_buffer += 50;
                }

                // --- Логика Пластичности (STDP Marking) ---

                // Таймеры тикают вниз
                if (syn.ltp_timer > 0) syn.ltp_timer--;
                if (syn.ltd_timer > 0) syn.ltd_timer--;

                // Событие: Pre-synaptic spike (Нейрон i выстрелил)
                if (neurons[i].spiked_this_step) {
                    // Pre выстрелил. Если Post (target) выстрелил НЕДАВНО - это LTP (Pre помог Post).
                    // Но в нашей каузальной схеме мы это проверим "наоборот" в следующем блоке.

                    // Здесь: Pre выстрелил. Мы ставим флаг "LTD ожидание". 
                    // Если Post НЕ выстрелит в ближайшее время, значит Pre стрелял впустую -> LTD.
                    syn.ltd_timer = TRACE_WINDOW;
                }

                // Событие: Post-synaptic spike (Цель выстрелила)
                if (neurons[syn.target_neuron_idx].spiked_this_step) {
                    // Post выстрелил.
                    // Если у синапса активен ltd_timer (значит Pre стрелял недавно),
                    // то это значит Pre -> Post. Это УСПЕХ (LTP).

                    // Внимание: мы перехватываем таймер LTD и превращаем его в LTP!
                    if (syn.ltd_timer > 0) {
                        syn.ltp_timer = TRACE_WINDOW; // Ставим метку на награду
                        syn.ltd_timer = 0;            // Убираем метку наказания
                    }
                }

                // --- Логика Обучения (R-STDP) ---
                if (reward) {
                    if (syn.ltp_timer > 0) {
                        // Награда + LTP флаг = Усиление уверенности
                        if (syn.confidence < CONFIDENCE_MAX) syn.confidence += 5;

                        //std::cout << "LEARNING! Synapse " << i << "->" << syn.target_neuron_idx
                          //  << " boosted to " << syn.confidence << std::endl;
                    }
                    else if (syn.ltd_timer > 0) {
                        // Награда + LTD флаг (Pre стрелял, а Post нет) = Это нам не помогло?
                        // Тут тонкий момент. Обычно наказывают за отсутствие награды.
                        // Но давай пока сделаем простое Hebbian learning с учителем.
                        // Пока оставим LTD только на "забывание".
                    }
                }

                // Обновление бинарного статуса
                syn.active = (syn.confidence >= CONFIDENCE_THR);
            }
        }
    }

    // Вспомогательное: получить количество спайков
    int count_spikes() {
        int c = 0;
        for (auto& n : neurons) if (n.spiked_this_step) c++;
        return c;
    }
};

// --- Main ---

int main() 
{
    srand(time(0));

    // 1. Инициализация
    int N = 100; // 100 нейронов
    SpikingNet brain(N);
    brain.build_random_topology(10); // По 10 связей на нейрон

    std::cout << "Neuron 0 is connected to: ";
    for (auto& syn : brain.connections[0]) {
        std::cout << syn.target_neuron_idx << " (conf=" << syn.confidence << ") ";
    }
    std::cout << std::endl;

    std::cout << "Simulating digital brain (" << N << " neurons)..." << std::endl;

    // 2. Цикл симуляции
    for (int t = 0; t < 10000; ++t) {

        // Входной сигнал (стимулируем нейроны 0, 1, 2 каждые 50 тактов)
        std::vector<int> inputs(N, 0);
        if (t % 50 < 5) {
            inputs[0] = 50; // Было 30
            inputs[1] = 50;
        }

        // Награда (условно: если нейрон 99 спайкнул, даем награду)
        bool reward = false;
        
        if (brain.neurons[99].spiked_this_step) {
            reward = true;
            std::cout << "GOAL REACHED at t=" << t << "!" << std::endl;
        }

        // Шаг
        brain.step(inputs, reward);

        // Лог
        int spikes = brain.count_spikes();
        if (spikes > 0) {
            std::cout << "t=" << t << " Spikes: " << spikes << " (Conf of syn 0->?: " << brain.connections[0][0].confidence << ")" << std::endl;
        }

        if (t % 100 == 0) { // Каждые 100 тактов, чтобы не спамить
            std::cout << "t=" << t << " Synapses from 0: ";
            for (auto& syn : brain.connections[0]) {
                std::cout << syn.target_neuron_idx << "=" << syn.confidence << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}