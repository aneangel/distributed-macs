#pragma once
#include <torch/torch.h>
#include "environment.h"

class RLAgent {
public:
    RLAgent(const std::vector<int>& architecture);
    torch::Tensor selectAction(const torch::Tensor& state);
    void update(const torch::Tensor& state, const torch::Tensor& action,
                const torch::Tensor& reward, const torch::Tensor& next_state, bool done);
    std::vector<torch::Tensor> getModelParameters();
    void setModelParameters(const std::vector<torch::Tensor>& parameters);

private:
    torch::nn::Sequential model;
    torch::optim::Adam optimizer;
    // Environment env;
};
