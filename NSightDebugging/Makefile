all:	solution

solution:	solution.o
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o solution -lm -lcuda -lrt solution.o ../libgputk/lib/libgputk.a

solution.o:	solution.cu
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o solution.o --compile solution.cu -I ../libgputk


dataset_generator:	dataset_generator.o
		nvcc -o dataset_generator -lm -lcuda -lrt dataset_generator.o ../libgputk/lib/libgputk.a 

dataset_generator.o:	dataset_generator.cpp
		nvcc --compile dataset_generator.cpp -I ../libgputk/

clean:
		@-rm -f dataset_generator.o
		@-rm -f dataset_generator
		@-rm -f template.o
		@-rm -f template
		@-rm -f solution.o
		@-rm -f solution
		@-rm -f a.out

run: solution
		@echo "\n"
		@echo "test - 0"
		@./solution -e ./Dataset/0/output.raw -i ./Dataset/0/input1.raw -t matrix
		@echo "\n"