CXX        := g++
CXXFLAGS   := -std=c++17 $(shell python3-config --includes)
LDFLAGS    := $(shell python3-config --ldflags --embed)
TARGET     := nexus_hic_bridge
SRCS      := main.cpp core/storage.cpp core/engine.cpp

.PHONY: all clean data run

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) $(LDFLAGS) -o $@

data:
	NEXUS_VENV="$${NEXUS_VENV:-./venv}" "$${NEXUS_VENV:-./venv}/bin/python" data/generate_data.py

run: $(TARGET) data
	./$(TARGET)

clean:
	rm -f $(TARGET) data/*.raw
