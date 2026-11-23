#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mpi.h"

#include "individual.h"

// --------------------------------------------------------
class GlobalHOF
{
public:
    // struct for serialized individuals to send with MPI
    struct SerializedEntry
    {
        double fitness = std::numeric_limits<double>::infinity();
        int size_W = 0;
        int size_Mo = 0;
        int depth_W = 0;
        int depth_Mo = 0;
        std::string tree_W;
        std::string tree_Mo;
    };

    std::vector<SerializedEntry> gatheredHof;  // gathered local HOFs in global hof

    static std::string SerializeEntry(const SerializedEntry& entry);
    static std::vector<char> SerializeLocalEntries(const std::vector<SerializedEntry>& localEntries);
    static std::vector<SerializedEntry> DeserializeLocalEntries(
        const std::vector<char>& buffer,
        size_t expectedCount
    );

    void GatherLocalHOF(
        int rank,
        int worldSize,
        int localEntryCount,
        const std::vector<Individual>& localHof
    );

    void BroadcastGlobalHOF(int rank);

    void writeGlobalHOF(int rank,
        const std::filesystem::path& outputDir = "output");  // root writes the gathered HOF
};


// ================================
std::string GlobalHOF::SerializeEntry(const SerializedEntry& entry)
{
    std::string buffer;
    buffer.reserve(
        sizeof(double) + 4 * sizeof(int) + 2 * sizeof(uint32_t) +
        entry.tree_W.size() + entry.tree_Mo.size()
    );

    auto appendValue = [&buffer](const auto& value)
    {
        const char* data = reinterpret_cast<const char*>(&value);
        buffer.append(data, sizeof(value));
    };

    appendValue(entry.fitness);
    appendValue(entry.size_W);
    appendValue(entry.size_Mo);
    appendValue(entry.depth_W);
    appendValue(entry.depth_Mo);

    const uint32_t lenW = static_cast<uint32_t>(entry.tree_W.size());
    appendValue(lenW);
    buffer.append(entry.tree_W);
    const uint32_t lenMo = static_cast<uint32_t>(entry.tree_Mo.size());
    appendValue(lenMo);
    buffer.append(entry.tree_Mo);

    return buffer;
}

// ================================
std::vector<char> GlobalHOF::SerializeLocalEntries(const std::vector<SerializedEntry>& localEntries)
{
    std::vector<char> buffer;
    size_t totalSize = 0;

    for (const auto& entry : localEntries)
    {
        const std::string serialized = SerializeEntry(entry);
        totalSize += serialized.size();
    }

    buffer.reserve(totalSize);

    for (const auto& entry : localEntries)
    {
        const std::string serialized = SerializeEntry(entry);
        buffer.insert(buffer.end(), serialized.begin(), serialized.end());
    }

    return buffer;
}

// ================================
std::vector<GlobalHOF::SerializedEntry> GlobalHOF::DeserializeLocalEntries(const std::vector<char>& buffer,
                                                                        size_t expectedCount)
{
    std::vector<SerializedEntry> entries;
    entries.reserve(expectedCount);

    size_t pos = 0;
    for (size_t i = 0; i < expectedCount; ++i)
    {
        SerializedEntry entry{};

        auto readValue = [&buffer, &pos](auto& value)
        {
            if (pos + sizeof(value) > buffer.size())
                throw std::runtime_error("Serialized HOF entry is truncated");

            std::memcpy(&value, buffer.data() + pos, sizeof(value));
            pos += sizeof(value);
        };

        readValue(entry.fitness);
        readValue(entry.size_W);
        readValue(entry.size_Mo);
        readValue(entry.depth_W);
        readValue(entry.depth_Mo);

        const auto readTree = [&]()
        {
            uint32_t len = 0;
            readValue(len);
            if (pos + len > buffer.size())
                throw std::runtime_error("Serialized HOF tree string is truncated");

            std::string tree(buffer.data() + pos, buffer.data() + pos + len);
            pos += len;
            return tree;
        };
        entry.tree_W = readTree();
        entry.tree_Mo = readTree();

        entries.push_back(entry);
    }

    return entries;
}


// ================================
void GlobalHOF::GatherLocalHOF(
    int rank,
    int worldSize,
    int localEntryCount,
    const std::vector<Individual>& localHof
)
{
    // ------------------------------------------
    // build the serialized local entries
    std::vector<SerializedEntry> localEntries(static_cast<size_t>(localEntryCount));

    for (int i = 0; i < localEntryCount; ++i)
    {
        if (i < static_cast<int>(localHof.size()))
        {
            const auto& ind = localHof[static_cast<size_t>(i)];
            localEntries[static_cast<size_t>(i)].fitness = ind.fitness; // save fitness

            auto& entry = localEntries[static_cast<size_t>(i)];
            entry.tree_W = ind.tree_W.convertToStr();
            entry.tree_Mo = ind.tree_Mo.convertToStr();
            entry.size_W = ind.tree_W.Size();
            entry.size_Mo = ind.tree_Mo.Size();
            entry.depth_W = ind.tree_W.Depth();
            entry.depth_Mo = ind.tree_Mo.Depth();
        }
        else 
        {
            throw std::runtime_error("Local HOF size is less than NlocalInds");
        }
    }

    // ------------------------------------------
    // gather the byte counts of the sendBuffer from all ranks to rank 0
    const std::vector<char> sendBuffer = SerializeLocalEntries(localEntries);   // serialized individuals to send
    const int sendCount = static_cast<int>(sendBuffer.size());

    std::vector<int> recvCounts(static_cast<size_t>(worldSize), 0);
    std::vector<int> recvEntryCounts(static_cast<size_t>(worldSize), 0);
    std::vector<int> displacements(static_cast<size_t>(worldSize), 0);

    MPI_Gather(
        &sendCount,         // const void *sendbuf
        1,                  // int sendcount
        MPI_INT,            // MPI_Datatype sendtype
        recvCounts.data(),  // void *recvbuf
        1,                  // int recvcount
        MPI_INT,            // MPI_Datatype recvtype
        0,                  // int root
        MPI_COMM_WORLD      // MPI_Comm comm
    );

    MPI_Gather(
        &localEntryCount,
        1,
        MPI_INT,
        recvEntryCounts.data(),
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    // ------------------------------------------
    // gather the data (now we know the size of each sendBuffer)
    if (rank == 0)
    {
        int totalRecv = 0;
        for (int i = 0; i < worldSize; ++i)
        {
            displacements[i] = totalRecv;
            totalRecv += recvCounts[i];
        }

        std::vector<char> recvBuffer(static_cast<size_t>(totalRecv));
        const char* sendPtr = sendCount > 0 ? sendBuffer.data() : nullptr;

        // MPI_Gatherv to send different sized buffers from each rank to rank 0
        MPI_Gatherv(
            const_cast<char*>(sendPtr),     // const void *sendbuf
            sendCount,                      // int sendcount
            MPI_CHAR,                       // MPI_Datatype sendtype
            recvBuffer.data(),              // void *recvbuf
            recvCounts.data(),              // const int recvcounts[]
            displacements.data(),           // const int displs[]
            MPI_CHAR,                       // MPI_Datatype recvtype
            0,                              // int root
            MPI_COMM_WORLD                  // MPI_Comm comm
        );

        size_t totalEntryCount = 0;
        for (const int entryCount : recvEntryCounts)
            totalEntryCount += static_cast<size_t>(entryCount);

        gatheredHof = DeserializeLocalEntries(recvBuffer, totalEntryCount);
    }
    else
    {
        const char* sendPtr = sendCount > 0 ? sendBuffer.data() : nullptr;
        MPI_Gatherv(
            const_cast<char*>(sendPtr), // const void *sendbuf
            sendCount,                  // int sendcount
            MPI_CHAR,                   // MPI_Datatype sendtype
            nullptr,                    // void *recvbuf
            nullptr,                    // const int recvcounts[]
            nullptr,                    // const int displs[]
            MPI_CHAR,                   // MPI_Datatype recvtype
            0,                          // int root
            MPI_COMM_WORLD              // MPI_Comm comm
        );
    }
}

// ================================
void GlobalHOF::BroadcastGlobalHOF(int rank)
{
    int entryCount = rank == 0 ? static_cast<int>(gatheredHof.size()) : 0;
    MPI_Bcast(
        &entryCount,    // void* buffer
        1,              // int count
        MPI_INT,        // MPI_Datatype datatype
        0,              // int root
        MPI_COMM_WORLD  // MPI_Comm comm
    );

    std::vector<char> buffer;
    int bufferSize = 0;
    if (rank == 0)
    {
        buffer = SerializeLocalEntries(gatheredHof);
        bufferSize = static_cast<int>(buffer.size());
    }

    MPI_Bcast(
        &bufferSize,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0)
        buffer.resize(static_cast<size_t>(bufferSize));

    MPI_Bcast(
        buffer.data(),
        bufferSize,
        MPI_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (rank != 0)
        gatheredHof = DeserializeLocalEntries(buffer, static_cast<size_t>(entryCount));
}

// ================================
void GlobalHOF::writeGlobalHOF(int rank, const std::filesystem::path& outputDir)
{
    // only rank 0 writes the globalHof file
    if (rank != 0)
        return;

    const std::filesystem::path outputDirMaster = std::filesystem::absolute(outputDir);
    std::filesystem::create_directories(outputDirMaster);

    const std::filesystem::path hofFilePathMaster = outputDirMaster / "global_hof.txt";
    std::ofstream hofFileGlobal(hofFilePathMaster, std::ios::trunc);
    if (!hofFileGlobal)
    {
        throw std::runtime_error("Failed to open output file: " + hofFilePathMaster.string());
    }

    int cntr = 1;   // counter for every individual
    for (const auto& entry : gatheredHof)
    {
        hofFileGlobal << "=========== " << cntr << " ===========\n"
            << "== Fitness : " << entry.fitness << "\n"
            << "== W Potential (Size : " << entry.size_W << ", Depth : " << entry.depth_W << ")" << "\n"
            << entry.tree_W << "\n"
            << "== Mo Potential (Size : " << entry.size_Mo << ", Depth : " << entry.depth_Mo << ")" << "\n"
            << entry.tree_Mo << "\n";

        cntr++;
    }

    hofFileGlobal.close();
}