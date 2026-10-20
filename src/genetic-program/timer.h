#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer
{
private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;
    std::string name;
    int rank = -1;  // used only for MPI processes

public:
    explicit Timer(const std::string& timerName)
        : 
        start(Clock::now()),
        name(timerName)
    {
    }

    explicit Timer(const std::string& timerName, const int rank)
        :
        start(Clock::now()),
        name(timerName),
        rank(rank)
    {
    }

    ~Timer()
    {
        auto end = Clock::now();

        double elapsed =
            std::chrono::duration<double>(end - start).count();

        std::cout << "==============================\n";

        if (rank != -1)
        {
            std::cout << "Rank: " << rank << "\n";
        }
        
        std::cout << name << ": "
                  << elapsed << " seconds.\n";
    }

    void setRank(int processRank)
    {
        rank = processRank;
    }
};