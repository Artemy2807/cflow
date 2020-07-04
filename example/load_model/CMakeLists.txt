cmake_minimum_required(VERSION 3.6)
# Имя проекта
project(example-load)
# Настройки CMake
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Флаги компиляции
if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -std=c++11 -Wall -g")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    add_compile_options(/W4 /WX-)
    if(NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.11)
        add_compile_options(/permissive-)
    endif()
endif()
# Если ОП Linux
if(UNIX AND NOT APPLE)
# Если не найдены файлы библиотеки для Linux
    # Если нет папки с библиотекой TensorFlow, то создаём
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()
    # Если нет файла библиотеки TensorFlow, то скачиваем из интернета библиотеку
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/libtensorflow.so)
        message(STATUS "Downloading tensorflow lib ...")
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-linux-x86_64-1.15.0.tar.gz ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/linux.tar.gz)
    endif()
    # Ищем файлы бибоитеки OpenCV
    find_package(OpenCV)
    # Подключаем файлы бибоитеки tensorflow и OpenCV
    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include ${OpenCV_INCLUDES})
    # Папка с файлами .lib для библиотеки tensorflow
    link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)
    # Копируем файл модели нейронной сети
    configure_file(../../models/model.pb ${CMAKE_CURRENT_BINARY_DIR}/model.pb COPYONLY)
    # Копируем картинку для тестового кода
    configure_file(../../img/to-test/img_20.jpg ${CMAKE_CURRENT_BINARY_DIR}/img_20.jpg COPYONLY)
# Если ОП Windows
elseif(WIN32)
    # Если нет папки с библиотекой TensorFlow, то создаём
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow)
        make_directory(${CMAKE_SOURCE_DIR}/tensorflow)
    endif()
    # Если нет файла библиотеки TensorFlow, то скачиваем из интернета библиотеку
    if (NOT EXISTS ${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll)
        message(STATUS "Downloading tensorflow lib ...")
        file(
        DOWNLOAD
            https://github.com/Neargye/tensorflow/releases/download/v1.15.0/libtensorflow-cpu-windows-x86_64-1.15.0.zip ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip
        STATUS
            status
        LOG
            log
        )
        list(GET status 0 status_code)
        list(GET status 1 status_string)
        if(NOT status_code EQUAL 0)
            message(FATAL_ERROR "error downloading tensorflow lib: ${status_string}" "${log}")
        endif()

        execute_process(COMMAND ${CMAKE_COMMAND} -E tar xvzf ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensorflow)
        file(REMOVE ${CMAKE_SOURCE_DIR}/tensorflow/windows.zip)
    endif()
    # Устаналиваем пути до библиотеки OpenCV
    # ВАЖНО укажите путь до вашей папки с собранной библиотекой OpenCV ВАЖНО
    set(OpenCV_DIR "C:/Program Files/opencv/build")
    set(OpenCV_INCLUDES ${OpenCV_DIR}/include)
    # Ищем файлы бибоитеки OpenCV
    find_package(OpenCV REQUIRED)
    # Копируем .dll файл библиотеки
    configure_file(${CMAKE_SOURCE_DIR}/tensorflow/lib/tensorflow.dll ${CMAKE_CURRENT_BINARY_DIR}/Debug/tensorflow.dll COPYONLY)
    # Подключаем файлы бибоитеки tensorflow
    include_directories(${CMAKE_SOURCE_DIR}/tensorflow/include ${OpenCV_INCLUDES})
    # Папка с файлами .lib для библиотеки tensorflow
    link_directories(${CMAKE_SOURCE_DIR}/tensorflow/lib)
    # Копируем файл модели нейронной сети
    configure_file(../../models/model.pb ${CMAKE_CURRENT_BINARY_DIR}/Debug/model.pb COPYONLY)
    # Копируем картинку для тестового кода
    configure_file(../../img/to-test/img_20.jpg ${CMAKE_CURRENT_BINARY_DIR}/Debug/img_20.jpg COPYONLY)
endif()
# Подключаем файлы бибоитеки для загрузки моделей
include_directories(../../include)
# Подключаем файлы программы для примера
aux_source_directory(../../src SOURCES)
set(SOURCES ${SOURCES}
    main.cpp
)
add_executable(example-load ${SOURCES})
# Добавляем файлы бибилотеки .lib tensorflow и OpenCV
target_link_libraries(example-load tensorflow ${OpenCV_LIBS})
