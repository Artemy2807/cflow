# CFLOW framework based on TensorFlow C API

![logo](logo/logo.png)

CFLOW - библиотека для облегчения работы с TensorFlow. Библиотека написана на c++. 

## [Example](/example)
Для просмотра более подробного описания примера нажмите на соответствующий пункт.
* [Load model](example/load_model/), загрузка модели нейронной сети и её использования в С++. 

Сборка на Windows

!ВАЖНО! Перед сборкой проект в файле [CMakeLists.txt](example/load_model/CMakeLists.txt), на строке set(OpenCV_DIR "C:/Program Files/opencv/build") под номером 42, укажите путь до вашей папки с собранной библиотекой OpenCV.
```cmd
git clone https://github.com/Artemy2807/cflow.git cflow
cd cflow/example/load_model/
mkdir build && cd build
cmake -A x64 ..
MSBuild.exe example-load.sln -property:Configuration=Debug
cd Debug
example-load
```

Сборка на Linux
```bash
git clone https://github.com/Artemy2807/cflow.git cflow
cd cflow/example/load_model/
mkdir build && cd build
cmake ..
make -j4
./example-load
```

* [Train CNN](example/train/), обучение нейронной сети на Keras и python.

## Contacts
Автор: Одышев Артемий
- Telegram: [@artemy](https://t.me/artemy_odeshev)
- VK: [@artemy](https://vk.com/artemyodiesiev)


 
