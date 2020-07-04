#include "model.hpp"

using namespace tf;


// Конструктор
Model::Model() {
    this->graph = TF_NewGraph();
	this->status = TF_NewStatus();
    TF_SessionOptions* options = TF_NewSessionOptions();
	this->session = TF_NewSession(graph, options, status);
    // Удаляем указатель
	TF_DeleteSessionOptions(options);
    if(TF_GetCode(status) != TF_OK) {
        printf("[ERROR]: create session is failed.\n");
        exit(EXIT_FAILURE);
    }
}

// Дескриптор
Model::~Model() {
    // Удаляем указатели
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

// Методы
int Model::load(const std::string& model_path) {
    FILE *file = fopen(model_path.c_str(), "rb");
	if (file == nullptr) {
		printf("[ERROR]: model %s not found.\n", model_path.c_str());
		exit(EXIT_FAILURE);
	}
    // Узнаём размер файла модели
	fseek(file, 0, SEEK_END);
	long fsize = ftell(file);
	fseek(file, 0, SEEK_SET);                                   
    // Считываем данные из файла модели
	void* data = malloc(fsize);
	fread(data, fsize, 1, file);
	fclose(file);
    if(data == nullptr) {
        printf("[ERROR]: model %s is empty.\n", model_path.c_str());
        exit(EXIT_FAILURE);
    }
	TF_Buffer* graph_buffer = TF_NewBufferFromString(data, fsize);
	TF_ImportGraphDefOptions* options = TF_NewImportGraphDefOptions();
	TF_GraphImportGraphDef(graph, graph_buffer, options, status);
	TF_DeleteImportGraphDefOptions(options);
	if (TF_GetCode(status) != TF_OK) {
		printf("[ERROR]: unable to import model %s.\n", TF_Message(status));
		exit(EXIT_FAILURE);
	}
	printf("Successfully imported model %s.\n", model_path.c_str());
	TF_DeleteBuffer(graph_buffer);
    // Узнаём имя входного и выходного параметра нейронной сети
    std::vector<std::string> operation_names = getOperation();
    this->input_name = operation_names[0];
    this->output_name = operation_names[operation_names.size() - 1];
    // Успешно завершили функцию
	return 0;
}

std::vector<float> Model::predict(const cv::Mat& input, const mat_info& info_input) {
    std::vector<float> output_vector;
    // Проверяем пустые ли входные данные
    if(input.empty() || this->graph == nullptr) return output_vector;
    cv::Mat transform;
    input.copyTo(transform);
    // Изменяем размер изображения
    if(info_input.size.first > 0 && info_input.size.second > 0) cv::resize(transform, transform, cv::Size(info_input.size.first, info_input.size.second));
    // Изменяем пространство цветов входного изображения
    switch(info_input.type) {
        case RGB:
        {
            #if CV_MAJOR_VERSION >= 3
            cv::cvtColor(transform, transform, cv::COLOR_BGR2RGB);
            #else
            cv::cvtColor(transform, transform, CV_BGR2RGB);
            #endif
            break;
        };
        case GRAY:
        {
            #if CV_MAJOR_VERSION >= 3
            cv::cvtColor(transform, transform, cv::COLOR_BGR2GRAY);
            #else
            cv::cvtColor(transform, transform, CV_BGR2GRAY);
            #endif
            break;
        };
    }
    // Конвертируем из cv::Mat в массив float
    std::vector<float> data = convert2Vector(transform);
    // Если задан флаг, то изменяем диапозон цвета пикселей
    if(info_input.range == BINARIZE)
        std::for_each(data.begin(), data.end(), [](float& p){ p /= (float)255.0;});
    // Создаём векторы с входными данными для нейронной сети в нужном формате
    std::vector<Tensor> output_tensors,
                        input_tensors;
    // Создаём структуру для входных данных нейронной сети
    Tensor mat_tensor(graph, this->input_name),
           output_tensor(graph, this->output_name);
    // Добавляем структуру в вектор для ответов нейронной сети
    output_tensors.push_back(output_tensor);
    // Добавляем входные данные в структуру
    mat_tensor.set(data);
    // Добавляем стрктуру для входных данных нейронной сети
    input_tensors.push_back(mat_tensor);
    // Используем нейронную сеть
    predict(input_tensors, output_tensors);
    // Получаем ответы нейронной сети в виде массива
    auto float_array = static_cast<float*>(TF_TensorData(output_tensors[0].tensor));
    // Получаем размер входных данных
    std::vector<int64_t> shape = output_tensors[0].shapeReturn();
    int count_element = std::accumulate(shape.begin(), shape.end(), 0);
    // Конвертируем массив в вектор для удобного использования
    output_vector = std::vector<float> {float_array, float_array + count_element};
    return output_vector;
}

void Model::predict(const std::vector<Tensor>& input, std::vector<Tensor>& output) {
    // Проверка входных данных
    if(std::all_of(input.begin(), input.end(), [](const Tensor& t){ return ((t.tensor == nullptr) || (t.operation.oper == nullptr)); })) {
        printf("[ERROR]: there are empty elements in the input vector.\n");
        exit(EXIT_FAILURE);
    }
    if(output.size() <= 0) {
        printf("[ERROR]: the size of the vector with the responses of the neural network must be greater than 0.\n");
        exit(EXIT_FAILURE);
    }
    // Заполняем вектора тензоров для входных и выходных данных
    std::vector<TF_Output> input_tf(input.size()), output_tf(output.size());
    std::transform(input.begin(), input.end(), input_tf.begin(), [](const Tensor& t){ return t.operation; });
    std::transform(output.begin(), output.end(), output_tf.begin(), [](const Tensor& t){ return t.operation; });
    // Заполняем вектор с данными для неронной сети
    std::vector<TF_Tensor*> tensors(input.size());
    std::transform(input.begin(), input.end(), tensors.begin(), [](const Tensor& t){ return t.tensor; });
    auto answer = new TF_Tensor*[output.size()];
    // Запускаем нейронную сеть
	TF_SessionRun(session,
        nullptr,
		input_tf.data(), tensors.data(), 1,
		output_tf.data(), answer, 1,
		nullptr, 0,
		nullptr,
		status
	);
    // Проверяем на ошибки
	if (TF_GetCode(status) != TF_OK) {
		printf("[ERROR]: run neural network is failed.\n");
        exit(EXIT_FAILURE);
	}
    // Заполняем вектор с ответами нейронной сети
    for(size_t i = 0; i < output.size(); i++) {
        output[i].tensor = answer[i];
    }
    // Удаляем указатель на ответы
    delete[] answer;
}

std::vector<std::string> Model::getOperation() {
    std::vector<std::string> names;
    if(graph == nullptr) return names;
    size_t position = 0;
    TF_Operation* operation;
    while((operation = TF_GraphNextOperation(graph, &position)) != nullptr)
        names.push_back(TF_OperationName(operation));
    return names;
}

std::vector<float> Model::convert2Vector(const cv::Mat& input) {
    // Конвертируем Mat в vector<float>
    std::vector<uint8_t> int_data;
	std::vector<float> float_data;
    if(input.empty()) return float_data;
    if(input.isContinuous())
        int_data.assign((uint8_t*)input.data, (uint8_t*)input.data + input.total() * input.channels());
    else 
        for(int i = 0; i < input.rows; i++)
            int_data.insert(int_data.end(), input.ptr<uint8_t>(i), input.ptr<uint8_t>(i) + input.cols * input.channels());
    float_data = std::vector<float>(int_data.begin(), int_data.end());
    return float_data;
}

