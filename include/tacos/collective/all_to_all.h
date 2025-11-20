/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Copyright (c) 2022-2025 Intel Corporation
Copyright (c) 2022-2025 Georgia Institute of Technology
*******************************************************************************/

#pragma once

#include <memory>
#include <tacos/collective/collective.h>
#include <tacos/topology/topology.h>

namespace tacos {

/// @brief All-to-All collective communication pattern
/// In All-to-All, each NPU has N chunks (one for each destination NPU)
/// Each NPU sends its i-th chunk to NPU i
class AllToAll final : public Collective {
  public:
    /// @brief Constructor for AllToAll collective
    /// @param npusCount number of NPUs in the topology
    /// @param collectivesCount number of rounds of all-to-all operations (default 1)
    explicit AllToAll(int npusCount, int collectivesCount = 1) noexcept;
    
    /// @brief Get the type of this collective
    [[nodiscard]] CollectiveType getType() const noexcept override { return CollectiveType::ALL_TO_ALL; }
};
}  // namespace tacos
