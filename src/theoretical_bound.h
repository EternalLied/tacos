/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/topology/topology.h>
#include <tacos/collective/collective.h>
#include <tacos/synthesizer/time_expanded_network.h>

namespace tacos {

/// @brief Calculate theoretical lower bound for collective time
/// @param topology Network topology
/// @param collective Collective operation
/// @param chunkSize Size of each chunk in bytes
/// @return Theoretical lower bound time in microseconds
double calculateTheoreticalLowerBound(const Topology& topology, 
                                     const Collective& collective,
                                     Collective::ChunkSize chunkSize);

}  // namespace tacos
