/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

// #define DEBUG
// #define ENABLE_PERF_STATS  // Comment this out to disable performance statistics

#ifdef DEBUG
#define DebugLog(x) do { x; } while(0)
#define ReleaseLog(x) do { } while(0)
#else
#define DebugLog(x) do { } while(0)
#define ReleaseLog(x) do { x; } while(0)
#endif

#ifdef ENABLE_PERF_STATS
#define PerfLog(x) do { x; } while(0)
#else
#define PerfLog(x) do { } while(0)
#endif
