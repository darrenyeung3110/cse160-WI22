bug1run:	bug1bin

bug1bin:	bug1.o
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o bug1bin -lm -lcuda -lrt bug1.o ../libgputk/lib/libgputk.a 

bug1.o:	bug1.cu
		nvcc -g -G -gencode arch=compute_60,code=sm_60 --compile bug1.cu -I ../libgputk/

bug2run:	bug2bin

bug2bin:	bug2.o
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o bug2bin -lm -lcuda -lrt bug2.o ../libgputk/lib/libgputk.a 

bug2.o:	bug2.cu
		nvcc -g -G -gencode arch=compute_60,code=sm_60 --compile bug2.cu -I ../libgputk/


bug3run: bug3bin

bug3bin: bug3.o
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o bug3bin -lm -lcuda -lrt bug3.o ../libgputk/lib/libgputk.a 

bug3.o:	bug3.cu
		nvcc -g -G -gencode arch=compute_60,code=sm_60 --compile bug3.cu -I ../libgputk/

bug4run: bug4bin

bug4bin: bug4.o
		nvcc -g -G -gencode arch=compute_60,code=sm_60 -o bug4bin -lm -lcuda -lrt bug4.o ../libgputk/lib/libgputk.a 

bug4.o:	bug4.cu
		nvcc -g -G -gencode arch=compute_60,code=sm_60 --compile bug4.cu -I ../libgputk/

clean:
		@-rm -f bug1.o
		@-rm -f bug1bin
		@-rm -f bug2.o
		@-rm -f bug2bin
		@-rm -f bug2.o
		@-rm -f bug2bin
		@-rm -f bug3.o
		@-rm -f bug3bin
		@-rm -f bug4.o
		@-rm -f bug4bin

run1: bug1bin
		@echo "\n"
		@echo "test - 0"
		@./bug1bin -e ./Dataset/0/output.raw -i ./Dataset/0/input0.raw,./Dataset/0/input1.raw -t matrix
		@echo "\n"
		@echo "test - 1"
		@./bug1bin -e ./Dataset/1/output.raw -i ./Dataset/1/input0.raw,./Dataset/1/input1.raw -t matrix
		@echo "\n"

run2: bug2bin
		@echo "\n"
		@echo "test - 0"
		@./bug2bin -e ./Dataset/0/output.raw -i ./Dataset/0/input0.raw,./Dataset/0/input1.raw -t matrix
		@echo "\n"
		@echo "test - 1"
		@./bug2bin -e ./Dataset/1/output.raw -i ./Dataset/1/input0.raw,./Dataset/1/input1.raw -t matrix
		@echo "\n"

run3: bug3bin
		@echo "\n"
		@echo "test - 0"
		@./bug3bin -e ./Dataset/0/output.raw -i ./Dataset/0/input0.raw,./Dataset/0/input1.raw -t matrix
		@echo "\n"
		@echo "test - 1"
		@./bug3bin -e ./Dataset/1/output.raw -i ./Dataset/1/input0.raw,./Dataset/1/input1.raw -t matrix
		@echo "\n"

run4: bug4bin
		@echo "\n"
		@echo "test - 0"
		@./bug4bin -e ./Dataset/0/output.raw -i ./Dataset/0/input0.raw,./Dataset/0/input1.raw -t matrix
		@echo "\n"