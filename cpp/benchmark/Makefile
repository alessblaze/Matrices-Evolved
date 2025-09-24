CXX = clang-20
CXXFLAGS = -std=c++23 -O3 -march=native -msse4.2 -mavx2 -Wall -Wextra -I./simde -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize -fsave-optimization-record
LDFLAGS = -lssl -lcrypto -lstdc++

all: full_bench

full_bench: full_bench.cpp functions.cpp
	$(CXX) $(CXXFLAGS) -o full_bench full_bench.cpp $(LDFLAGS)

run: full_bench
	./full_bench

clean:
	rm -f full_bench

.PHONY: all clean run