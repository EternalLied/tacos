/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief Build AMD MI250 two chassis topology
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildAMD_MI250_2Chassis(Topology& topo, bool allow_copy = false);

/// @brief Build AMD MI250 four chassis topology
/// @param topo Topology object to build into
/// @param allow_copy Whether to allow copy operations
void BuildAMD_MI250_4Chassis(Topology& topo, bool allow_copy = false);

}  // namespace tacos
