CXX = g++
CXXFLAGS = --std=c++11 -Wall -Wno-sign-compare -Wno-unknown-pragmas -fPIC -fopenmp -O3 -pthread
INCLUDE = -I"./submodules/cxxopts/include"

scne : scne.o lossycounting.o src/main.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) scne.o lossycounting.o src/main.cpp -o scne

scne.o : src/cheaprand.h src/scne.h src/scne.cpp
	$(CXX) $(CXXFLAGS) -c src/scne.cpp -o scne.o

lossycounting.o : src/lossycounting.h src/lossycounting.cpp
	$(CXX) $(CXXFLAGS) -c src/lossycounting.cpp -o lossycounting.o

clean:
	rm -f -r *.o scne
