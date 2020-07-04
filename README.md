# CFLOW framework based on TensorFlow C API

![logo](logo/logo.png)

CFLOW - библиотека для облегчения работы с TensorFlow. Библиотека написана на c++. 

## Official Builds
Branch | Linux | Windows |
-------|:-----:|:-------:|
master |[![Build Status](https://travis-ci.org/Neargye/hello_tf_c_api.svg)]()|[![Build status](https://travis-ci.org/Neargye/hello_tf_c_api.svg?branch=master)]()

## [Example](/example)
Для просмотра более подробного описания примера нажмите на соответствующий пункт.
* [Load model](example/load_model/), загрузка модели нейронной сети и её использования в С++. 

Сборка на Windows
!ВАЖНО! Перед сборкой проект в файле [CMakeLists.txt](example/load_model/CMakeLists.txt), на строке set(OpenCV_DIR "C:/Program Files/opencv/build") под номером 42, укажите путь до вашей папки с собранной библиотекой OpenCV.
```cmd
git clone https://github.com/Artemy2807/cflow.git
cd cflow/github/example/load_model/
mkdir build && cd build
cmake -A x64 ..
MSBuild.exe example-load.sln -property:Configuration=Debug
cd Debug
example-load
```

Сборка на Linux
```bash
git clone https://github.com/Artemy2807/cflow.git
cd cflow/github/example/load_model/
mkdir build && cd build
cmake ..
make -j4
./example-load
```

## Contacts
Автор: Одышев Артемий
- Telegram: [@artemy](https://t.me/artemy_odeshev)
- VK: [@artemy](https://vk.com/artemyodiesiev)


 
