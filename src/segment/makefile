CPPFLAGS = -I ../../install/include -D __STDC_CONSTANT_MACROS

all: segment

segment: main.o LocalHistCmp.o
	g++ $$(pkg-config --with-path=../../install/lib/pkgconfig --libs libavformat libavcodec libavutil libswscale) -o segment main.o LocalHistCmp.o

main.o: main.cpp LocalHistCmp.hpp
	g++ $(CPPFLAGS) -std=c++17 -c main.cpp

LocalHistCmp.o: LocalHistCmp.cpp LocalHistCmp.hpp
	g++ $(CPPFLAGS) -c LocalHistCmp.cpp

clean:
	rm segment main.o LocalHistCmp.o
