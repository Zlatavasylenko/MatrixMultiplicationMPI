#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int ProcNum = 0; // Кількість доступних процесів
int ProcRank = 0; // Ранг поточного процесу

// Функція для ініціалізації матриць випадковими числами від 0 до 9
void RandomDataInitialization(double* pMatrixA, double* pMatrixB, int Size) {
    if (ProcRank == 0) {
        srand(unsigned(time(NULL)));
        for (int i = 0; i < Size * Size; i++) {
            pMatrixA[i] = rand() % 10; // Випадкові числа від 0 до 9
            pMatrixB[i] = rand() % 10;
        }
    }
}

// Функція для запису матриці у файл (тільки процес 0)
void WriteMatrixToFile(double* pMatrix, int Size, const char* filename) {
    if (ProcRank == 0) {
        FILE* file = nullptr;
        errno_t err = fopen_s(&file, filename, "w");
        if (err != 0 || file == nullptr) {
            printf("Помилка: Неможливо відкрити файл %s для запису.\n", filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < Size; i++) {
            for (int j = 0; j < Size; j++) {
                fprintf(file, "%7.1f ", pMatrix[i * Size + j]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

// Функція для послідовного множення матриць (для перевірки)
void SerialMatrixMultiplication(double* pMatrixA, double* pMatrixB, double* pMatrixC, int Size) {
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            pMatrixC[i * Size + j] = 0;
            for (int k = 0; k < Size; k++) {
                pMatrixC[i * Size + j] += pMatrixA[i * Size + k] * pMatrixB[k * Size + j];
            }
        }
    }
}

// Функція для перевірки коректності результатів
void VerifyResult(double* pMatrixA, double* pMatrixB, double* pMatrixC, int Size) {
    if (ProcRank == 0) {
        double* pSerialMatrixC = new double[Size * Size];
        SerialMatrixMultiplication(pMatrixA, pMatrixB, pSerialMatrixC, Size);
        bool correct = true;
        for (int i = 0; i < Size * Size; i++) {
            if (pMatrixC[i] != pSerialMatrixC[i]) {
                correct = false;
                break;
            }
        }
        if (correct) {
            printf("Паралельні та послідовні результати збігаються.\n");
        }
        else {
            printf("Помилка: Паралельні та послідовні результати відрізняються!\n");
        }
        delete[] pSerialMatrixC;
    }
}

// Функція для ініціалізації даних та виділення пам’яті
void ProcessInitialization(double*& pMatrixA, double*& pMatrixB, double*& pMatrixC,
    double*& pProcRowsA, double*& pProcResultC, int& Size, int& RowNum, int argc, char* argv[]) {
    setvbuf(stdout, 0, _IONBF, 0);
    if (ProcRank == 0) {
        if (argc > 1) {
            Size = atoi(argv[1]);
        }
        else {
            printf("\nВведіть розмір матриць (наприклад, 100): ");
            if (scanf_s("%d", &Size) != 1 || Size <= 0) {
                printf("Некоректне введення! Розмір має бути додатним цілим числом.\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Розподіл рядків між процесами
    int baseRows = Size / ProcNum; // Базова кількість рядків на процес
    int extraRows = Size % ProcNum; // Залишкові рядки
    if (ProcRank < extraRows) {
        RowNum = baseRows + 1; // Перші процеси отримують на один рядок більше
    }
    else {
        RowNum = baseRows; // Решта отримує базову кількість
    }

    // Виділення пам’яті
    pMatrixB = new double[Size * Size]; // Повна матриця B на кожному процесі
    pMatrixC = new double[Size * Size]; // Результуюча матриця C
    pProcRowsA = new double[RowNum * Size]; // Смуга матриці A
    pProcResultC = new double[RowNum * Size]; // Смуга результуючої матриці C

    if (ProcRank == 0) {
        pMatrixA = new double[Size * Size]; // Повна матриця A
        RandomDataInitialization(pMatrixA, pMatrixB, Size);
        WriteMatrixToFile(pMatrixA, Size, "matrixA.txt");
        WriteMatrixToFile(pMatrixB, Size, "matrixB.txt");
    }
    else {
        pMatrixA = nullptr;
    }
}

// Функція для розподілу матриці A та розсилки матриці B
void DataDistribution(double* pMatrixA, double* pMatrixB, double* pProcRowsA, int Size, int RowNum) {
    int* pSendNum = new int[ProcNum]; // Кількість елементів для кожного процесу
    int* pSendInd = new int[ProcNum]; // Індекси для розподілу
    int baseRows = Size / ProcNum;
    int extraRows = Size % ProcNum;

    // Обчислення кількості елементів та зміщень
    pSendInd[0] = 0;
    for (int i = 0; i < ProcNum; i++) {
        pSendNum[i] = (i < extraRows ? baseRows + 1 : baseRows) * Size;
        if (i > 0) {
            pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
        }
    }

    MPI_Scatterv(pMatrixA, pSendNum, pSendInd, MPI_DOUBLE, pProcRowsA,
        RowNum * Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pMatrixB, Size * Size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
}

// Функція для паралельного обчислення результуючої матриці
void ParallelResultCalculation(double* pProcRowsA, double* pMatrixB, double* pProcResultC, int Size, int RowNum) {
    for (int i = 0; i < RowNum; i++) {
        for (int j = 0; j < Size; j++) {
            pProcResultC[i * Size + j] = 0;
            for (int k = 0; k < Size; k++) {
                pProcResultC[i * Size + j] += pProcRowsA[i * Size + k] * pMatrixB[k * Size + j];
            }
        }
    }
}

// Функція для збору результуючої матриці C
void ResultReplication(double* pProcResultC, double* pMatrixC, int Size, int RowNum) {
    int* pReceiveNum = new int[ProcNum]; // Кількість елементів для збору
    int* pReceiveInd = new int[ProcNum]; // Індекси для збору
    int baseRows = Size / ProcNum;
    int extraRows = Size % ProcNum;

    pReceiveInd[0] = 0;
    for (int i = 0; i < ProcNum; i++) {
        pReceiveNum[i] = (i < extraRows ? baseRows + 1 : baseRows) * Size;
        if (i > 0) {
            pReceiveInd[i] = pReceiveInd[i - 1] + pReceiveNum[i - 1];
        }
    }

    MPI_Gatherv(pProcResultC, RowNum * Size, MPI_DOUBLE, pMatrixC,
        pReceiveNum, pReceiveInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pReceiveNum;
    delete[] pReceiveInd;
}

// Функція для завершення роботи та звільнення пам’яті
void ProcessTermination(double* pMatrixA, double* pMatrixB, double* pMatrixC,
    double* pProcRowsA, double* pProcResultC) {
    if (ProcRank == 0) {
        delete[] pMatrixA;
    }
    delete[] pMatrixB;
    delete[] pMatrixC;
    delete[] pProcRowsA;
    delete[] pProcResultC;
}

int main(int argc, char* argv[]) {
    double* pMatrixA; // Перша матриця
    double* pMatrixB; // Друга матриця
    double* pMatrixC; // Результуюча матриця
    int Size; // Розмір матриць (N x N)
    double* pProcRowsA; // Смуга матриці A
    double* pProcResultC; // Смуга результуючої матриці C
    int RowNum; // Кількість рядків у смузі
    double Start, Finish, Duration;

    MPI_Init(&argc, &argv); // Ініціалізація MPI
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum); // Визначення кількості процесів
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank); // Визначення рангу процесу

    if (ProcRank == 0) {
        printf("Програма для паралельного множення матриць (блочний алгоритм)\n");
    }

    // Ініціалізація даних
    ProcessInitialization(pMatrixA, pMatrixB, pMatrixC, pProcRowsA, pProcResultC, Size, RowNum, argc, argv);

    // Вимірювання часу
    Start = MPI_Wtime();
    DataDistribution(pMatrixA, pMatrixB, pProcRowsA, Size, RowNum);
    ParallelResultCalculation(pProcRowsA, pMatrixB, pProcResultC, Size, RowNum);
    ResultReplication(pProcResultC, pMatrixC, Size, RowNum);
    Finish = MPI_Wtime();
    Duration = Finish - Start;

    // Вивід результатів
    if (ProcRank == 0) {
        VerifyResult(pMatrixA, pMatrixB, pMatrixC, Size);
        WriteMatrixToFile(pMatrixC, Size, "matrixC.txt");
        printf("Кількість процесів: %d, Час виконання: %f секунд\n", ProcNum, Duration);
    }

    // Завершення роботи
    ProcessTermination(pMatrixA, pMatrixB, pMatrixC, pProcRowsA, pProcResultC);
    MPI_Finalize();
    return 0;
}