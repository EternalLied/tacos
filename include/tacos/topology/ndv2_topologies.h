/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief Build NDv2 two chassis topology (TE-CCL version)
/// @param topo Topology object to build into
void BuildNDv2_TwoChassis_Tecc(Topology& topo);

/// @brief Build NDv2 four chassis topology (TE-CCL version)
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildNDv2_FourChassis_Tecc(Topology& topo, bool allow_copy = false);

/// @brief Build NDv2 two chassis topology
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildNDv2_TwoChassis(Topology& topo, bool allow_copy = false);

/// @brief Build NDv2 four chassis topology
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildNDv2_FourChassis(Topology& topo, bool allow_copy = false);

}  // namespace tacos
