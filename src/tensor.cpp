#include "tensor.hpp"

using namespace tf;

// Конструктор
Tensor::Tensor(TF_Graph* graph, const std::string& operation_name) {
	operation.oper = TF_GraphOperationByName(graph, operation_name.c_str());
	operation.index = 0;
	if (operation.oper == nullptr) {
		printf("[ERROR]: no operation named %s find.\n", operation_name.c_str());
		exit(EXIT_FAILURE);
	}
	TF_Status* status = TF_NewStatus();
	int number_dims = TF_GraphGetTensorNumDims(graph, operation, status);
	this->type = TF_OperationOutputType(operation);
	if (number_dims > 0) {
		int64_t *dims = new int64_t[number_dims];
		TF_GraphGetTensorShape(graph, operation, dims, number_dims, status);
		if (TF_GetCode(status) != TF_OK) {
			printf("[ERROR]: get tensor shape is failed.\n");
			exit(EXIT_FAILURE);
		}
		this->shape = std::vector<int64_t>(dims, dims + number_dims);
		delete[] dims;
	}
    TF_DeleteStatus(status);
	this->data = nullptr;
    this->tensor = nullptr;
}

// Дескриптор
Tensor::~Tensor() {
	this->tensor = nullptr;
    this->data = nullptr;
}

// Методы
void Tensor::clean() {
	if(this->tensor != nullptr) TF_DeleteTensor(this->tensor);
}

void Tensor::set(const std::vector<float>& data_input) {
    if (data_input.size() <= 0) {
		printf("[ERROR]: empty data in function set.\n");
		exit(EXIT_FAILURE);
	}
	if (type != TF_FLOAT) {
		printf("[ERROR]: provided type is different from Tensor expected type.\n");
		exit(EXIT_FAILURE);
	}
    if(tensor != nullptr) TF_DeleteTensor(tensor);
    // Лямбда функция для удаления указателя
	auto dealloator = [](void* ddata, size_t, void*) {  };
    int64_t exp_size = std::abs(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()));
	std::unique_ptr<std::vector<int64_t>> actual_shape = std::make_unique<std::vector<int64_t>>(shape.begin(), shape.end());
	std::replace_if(actual_shape->begin(), actual_shape->end(), [](int64_t r) { return r == -1; }, data_input.size() / exp_size);
    // Копируем данные из массива в общий указатель
	this->data = malloc(sizeof(float) * data_input.size());
	float* to_float = const_cast<float*>(data_input.data());
	this->data = static_cast<void*>(to_float);
	actual_shape->data()[3] = 3;
	this->tensor = TF_NewTensor(type, actual_shape->data(), actual_shape->size(), this->data, sizeof(float) * data_input.size(), dealloator, nullptr);
	if (tensor == nullptr) {
		printf("[ERROR]: an error occurred allocating the Tensor memory.\n");
		exit(EXIT_FAILURE);
	}
	this->data = nullptr;
}

std::vector<int64_t> Tensor::shapeReturn() {
    std::vector<int64_t> shape_actual = shape;
    shape_actual.erase(shape_actual.begin(), shape_actual.begin() + 1);
    return shape_actual;
}