#pragma once
#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>
#include "tensor.hpp"
#include <string>

namespace opencv {
    enum mat_type {
        NONE = 0,
        RGB,
        GRAY,
    };

    enum pixel_range {
        STANDART = 0,
        BINARIZE,
    };

    struct mat_info {
        mat_type type;
        std::pair<int, int> size;
        pixel_range range;
    };
}

namespace tf {
    using namespace opencv;
    class Model {
    private:
        TF_Graph* graph;
        TF_Status* status;
        TF_Session* session;
        std::string input_name, output_name;
    private:
        std::vector<float> convert2Vector(const cv::Mat& input);
    public:
        Model();
        ~Model();
        int load(const std::string& model_path);
        std::vector<std::string> getOperation();
        std::vector<float> predict(const cv::Mat& input, const mat_info& info_input);
        void predict(const std::vector<Tensor>& input, std::vector<Tensor>& output);
    };
}