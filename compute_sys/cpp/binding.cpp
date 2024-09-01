#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include "rl_agent.h"

namespace py = pybind11;

PYBIND11_MODULE(rl_cpp, m) {
    py::class_<RLAgent>(m, "RLAgent")
        .def(py::init<const std::vector<int>&>())
        .def("select_action", &RLAgent::selectAction)
        .def("update", &RLAgent::update)
        .def("get_model_parameters", &RLAgent::getModelParameters)
        .def("set_model_parameters", &RLAgent::setModelParameters);
}