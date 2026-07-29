cmake_minimum_required(VERSION 3.20)
project(Media_Player)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_AUTOMOC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia MultimediaWidgets)

if(WIN32)
    add_executable(Media_Player WIN32
        src/main.cpp
        src/TimelineBar.cpp
        include/TimelineBar.h
    )
elseif(APPLE)
    add_executable(Media_Player MACOSX_BUNDLE
        src/main.cpp
        src/TimelineBar.cpp
        include/TimelineBar.h
    )
else()
    add_executable(Media_Player
        src/main.cpp
        src/TimelineBar.cpp
        include/TimelineBar.h
    )
endif()

target_include_directories(Media_Player PRIVATE include)

target_link_libraries(Media_Player
    Qt6::Widgets
    Qt6::Multimedia
    Qt6::MultimediaWidgets
)