#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "xyz-parser.h"

// header file for reading POSCAR files for 1 element potential
// for 2 element check xyz-parser.h


// cutoff distance
constexpr double cutoff = 5.0;


// =============================================================
struct PoscarData {
    double scale = 0.0;
    Vec3 lattice[3];    // lattice vectors (also the dimensions of pbc box)
    int numAtoms = 0;
    bool directCoordinates = false;
    std::vector<Vec3> positions;    // cartesian coordinates
    double energyReal = 0.0;
};


// =============================================================
struct PairDistanceData
{
    std::vector<double> distances;
    double energyReal;
};

// =============================================================
PoscarData readPOSCAR(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open POSCAR");
    }

    PoscarData data;

    std::string line;

    // Comment
    std::getline(file, line);

    // Scale
    file >> data.scale; // in those data Scale = 1 !

    // Lattice vectors
    for (int i = 0; i < 3; ++i) 
    {
        file >> data.lattice[i].x
            >> data.lattice[i].y
            >> data.lattice[i].z;
    }

    // Atomic species
    std::getline(file, line);
    std::getline(file, line);

    // Number of atoms
    file >> data.numAtoms;

    // Coordinate type
    std::getline(file, line);
    std::getline(file, line);

    // check if direct coordinates
    data.directCoordinates = (line[0] == 'D' || line[0] == 'd');
    if (!data.directCoordinates) {
        throw std::runtime_error("Not direct coordinates !");
    }


    // Positions ( we dont rescale, scale = 1)
    data.positions.resize(data.numAtoms);


    for (int i = 0; i < data.numAtoms; ++i)
    {
        double x, y, z;

        file >> x >> y >> z;

        // Direct to Cartesian
        data.positions[i].x =
            x * data.lattice[0].x +
            y * data.lattice[1].x +
            z * data.lattice[2].x;

        data.positions[i].y =
            x * data.lattice[0].y +
            y * data.lattice[1].y +
            z * data.lattice[2].y;

        data.positions[i].z =
            x * data.lattice[0].z +
            y * data.lattice[1].z +
            z * data.lattice[2].z;
    }

    return data;
}

// =============================================================
std::vector<double> readEnergies(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open energy file: " + filename);
    }

    std::vector<double> energies;
    double energy;

    while (file >> energy)
    {
        energies.push_back(energy);
    }

    return energies;
}

// =============================================================
std::vector<PoscarData> storeData(std::string pathName, int snapshotSize)
{
    std::vector<PoscarData> dataVector;
    std::vector<double> Energies = readEnergies(pathName + "E0.data");

    for (int i = 1; i <= snapshotSize; i++)
    {
        std::string filename = pathName +
            std::to_string(i) +
            ".vasp";

        PoscarData data = readPOSCAR(filename);

        data.energyReal = Energies[i - 1];	// store snapshot energy

        dataVector.push_back(data);
    }

    return dataVector;
}

// =============================================================
double pbcDistance(const Vec3& pos1, const Vec3& pos2, const PoscarData& snapshot)
{
    double dx = pos1.x - pos2.x;
    double dy = pos1.y - pos2.y;
    double dz = pos1.z - pos2.z;

    // (the lattice is cubic so only the diagonal is non-zero)
    double Lx = snapshot.lattice[0].x;
    double Ly = snapshot.lattice[1].y;
    double Lz = snapshot.lattice[2].z;

    // pbc
    dx -= Lx * std::round(dx / Lx);
    dy -= Ly * std::round(dy / Ly);
    dz -= Lz * std::round(dz / Lz);

    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// =============================================================
std::vector<std::vector<PairDistanceData>> PrecomputeDistances(
    const std::vector<std::vector<PoscarData>>& data)
{
    std::vector<std::vector<PairDistanceData>> precomputed;

    precomputed.reserve(data.size());

    for (const auto& simulation : data)
    {
        std::vector<PairDistanceData> simulationData;
        simulationData.reserve(simulation.size());

        for (const auto& snapshot : simulation)
        {
            PairDistanceData snapshotData;

            const auto& positions = snapshot.positions;
            const size_t N = positions.size();

            // Loop over unique atom pairs
            for (size_t i = 0; i < N; ++i)
            {
                for (size_t j = i + 1; j < N; ++j)
                {
                    double r = pbcDistance(positions[i], positions[j], snapshot);

                    if (r >= cutoff)    // skip large r
                        continue;

                    snapshotData.distances.push_back(r);
                }
            }

            snapshotData.energyReal = snapshot.energyReal;  // store E snapshot

            simulationData.push_back(std::move(snapshotData));
        }

        precomputed.push_back(std::move(simulationData));
    }

    return precomputed;
}