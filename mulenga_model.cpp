#include "model.h"
#include <cmath>
#include <fstream>
#include <random>
#include <algorithm>

MulengaLM::MulengaLM(int vocab, int hidden)
       : vocab_size(vocab), hidden_size(hidden),
         Wxh(hidden, std::vector<float>(vocab)),
         Why(vocab, std::vector<float>(hidden)),
         bh(hidden, 0.0f),
         by(vocab, 0.0f),
         h(hidden, 0.0f) {}

int MulengaLM::char_to_index(char c) {
    return c - 'a';
}

char MulengaLM::index_to_char(int i) {
    return 'a' + i;
}
std::vector<float> MulengaLM::softmax(const std::vector<float>& x) {
    std::vector<float> result(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];

    }
    for (float& v : result) v /= sum;
    return result;
}

char MulengaLM::predict(char input_char) {
    int x_idx = char_to_index(input_char);

    // Hidden layer
    for (int i = 0; i < hidden_size; i++) {
        h[i] = std::tanh(Wxh[i][x_idx] + bh[i]);
    }
    // Output layer
    std::vector<float> logits(vocab_size);
    for (int i = 0; i <  vocab_size; i++) {
        logits[i] = by[i];
        for (int j = 0; j < hidden_size; j++) {
            logits[i] += Why[i][j] * h[j];
        }
    }

    auto probs = softmax(logits);

    // sample
    static std::mt19937 gen(std::random_device{}());
    std::discrete_distribution<int> dist(probs.begin(), probs.end());

    return index_to_char(dist(gen));
}
std::string MulengaLM::generate(char seed, int length) {
    std::string output;
    char current = seed;
    output += current;

    for (int i = 0; i < length; i++) {
        current =  predict(current);
        output += current;
    }
    return output;
}
