#pragma once
#include "tensorflow/c/c_api.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <memory>
#include <vector>
#include <string>

namespace tf {
    class Tensor {
    private:
        std::vector<int64_t> shape;
        TF_DataType type;
        void* data;
    public:
        TF_Tensor* tensor;
        TF_Output operation;
    public:
        Tensor(TF_Graph* graph, const std::string& operation_name);
        ~Tensor();
        void clean();
        void set(const std::vector<float>& data_input);
        std::vector<int64_t> shapeReturn();
    };
}