#include "rl_agent.h"

RLAgent::RLAgent(const std::vector<int>& architecture) {
    for (size_t i = 0; i < architecture.size() - 1; ++i) {
        model->push_back(torch::nn::Linear(architecture[i], architecture[i + 1]));
        if (i < architecture.size() - 2) {
            model->push_back(torch::nn::ReLU());
        }
    }
    optimizer = torch::optim::Adam(model->parameters());
}

torch::Tensor RLAgent::selectAction(const torch::Tensor& state) {
    torch::NoGradGuard no_grad;
    return model->forward(state).argmax();
}

void RLAgent::update(const torch::Tensor& state, const torch::Tensor& action,
                     const torch::Tensor& reward, const torch::Tensor& next_state, bool done) {
    // Implement your RL algorithm here (e.g., DQN, PPO)
}

std::vector<torch::Tensor> RLAgent::getModelParameters() {
    std::vector<torch::Tensor> params;
    for (const auto& param : model->parameters()) {
        params.push_back(param.clone());
    }
    return params;
}

void RLAgent::setModelParameters(const std::vector<torch::Tensor>& parameters) {
    auto params = model->parameters();
    for (size_t i = 0; i < params.size(); ++i) {
        params[i].data().copy_(parameters[i]);
    }
}