/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief Build DGX1 topology (TE-CCL version)
/// @param topo Topology object to build into
/// @param alpha_us Latency in microseconds
void BuildDGX1(Topology& topo, double alpha_us = 0.7);

/// @brief Build DGX2 two chassis topology (TE-CCL version)
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildDGX2_TwoChassis(Topology& topo, bool allow_copy = false);

/// @brief Build DGX2 two chassis topology - Type A (single shared IB switch, 16 ports)
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildDGX2_TwoChassis_typeA(Topology& topo, bool allow_copy = false);

/// @brief Build DGX2 two chassis topology - Type B (dual IB switches, 8 ports each)
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildDGX2_TwoChassis_typeB(Topology& topo, bool allow_copy = false);

}  // namespace tacos
