/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <tacos/topology/topology.h>
#include <string>
#include <optional>

namespace tacos {

/// @brief Create a topology by name
/// @param topologyName Name of the topology to create
/// @return Topology object, or nullopt if topology name is invalid
std::optional<Topology> createTopology(const std::string& topologyName);

/// @brief Get list of available topology names
/// @return String containing all available topology names
std::string getAvailableTopologies();

}  // namespace tacos
