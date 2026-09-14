// core_affinity.cpp
// Day 21 Update: Core affinity logic is fully header-only in core_affinity.h, 
// now including cross-platform support for both Windows (SetThreadAffinityMask) 
// and Linux (sched_setaffinity).
// This file is kept to satisfy build system conventions or for future 
// non-inline OS-specific expansions.
#include "core_affinity.h"