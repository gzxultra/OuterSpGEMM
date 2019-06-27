RMATPATH = GTgraph/R-MAT
SPRNPATH = GTgraph/sprng2.0-lite

include GTgraph/Makefile.var
INCLUDE += -I$(SPRNPATH)/include
CC = g++

FLAGS = -fopenmp

sprng:
	(cd $(SPRNPATH); $(MAKE); cd ../..)

rmat:	sprng
	(cd $(RMATPATH); $(MAKE); cd ../..)

TOCOMPILE = $(RMATPATH)/graph.o $(RMATPATH)/utils.o $(RMATPATH)/init.o $(RMATPATH)/globals.o

# flags defined in GTgraph/Makefile.var
SAMPLE = ./sample
BIN = ./bin
SRC_SAMPLE = $(wildcard $(SAMPLE)/*.cpp)
SAMPLE_TARGET = $(SRC_SAMPLE:$(SAMPLE)%=$(BIN)%)

spgemm: rmat $(SAMPLE_TARGET:.cpp=_hw)

$(BIN)/%_hw: $(SAMPLE)/%.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP -DHW_EXE ${TOCOMPILE} ${LIBS}

# specific for OuterSpGEMM
# Will do the same for other exe files
$(BIN)/OuterSpGEMM_hw: outer_mult.h sample/OuterSpGEMM.cpp
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(INCLUDE) -o $@ $^ -DCPP -DHW_EXE ${TOCOMPILE} ${LIBS}

clean:
	(cd GTgraph; make clean; cd ../..)
	rm -rf ./bin/*
	rm -rf assets/*


gen-er:
	./scripts/gen_er.sh

gen-rmat:
	./scripts/gen_rmat.sh

download:
	./scripts/download.sh
